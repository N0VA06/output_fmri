import os
import sys
import json
import time
import logging
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import numpy as np

try:
    from bids import BIDSLayout
except ImportError:
    print("Please install pybids: pip install pybids")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fmriprep_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScannerConfig:
    field_strength: str
    bold2anat_dof: int
    bold2anat_init: str
    use_syn_sdc: bool
    output_spaces: List[str]
    slice_time_ref: float
    fd_spike_threshold: float
    dvars_spike_threshold: float

    @classmethod
    def from_session_id(cls, session_id: str) -> 'ScannerConfig':
        configs = {
            '1.5T': cls(
                field_strength='1.5T',
                bold2anat_dof=6,
                bold2anat_init='register',  # header or register
                use_syn_sdc=False,
                output_spaces=['MNI152NLin2009cAsym:res-2', 'T1w'],
                slice_time_ref=0.5,
                fd_spike_threshold=0.5,
                dvars_spike_threshold=1.5
            ),
            '3T': cls(
                field_strength='3T',
                bold2anat_dof=6,
                bold2anat_init='register',
                use_syn_sdc=False,
                output_spaces=['MNI152NLin2009cAsym:res-2', 'T1w', 'fsaverage5'],
                slice_time_ref=0.5,
                fd_spike_threshold=0.4,
                dvars_spike_threshold=1.3
            ),
            '7T': cls(
                field_strength='7T',
                bold2anat_dof=9,
                bold2anat_init='header',  
                use_syn_sdc=True,
                output_spaces=[
                    'MNI152NLin2009cAsym:res-2',
                    'T1w',
                    'fsLR:32k',
                    'fsaverage5'
                ],
                slice_time_ref=0.5,
                fd_spike_threshold=0.3,
                dvars_spike_threshold=1.2
            )
        }
        for key, cfg in configs.items():
            if key in session_id:
                return cfg
        # Default fallback to 3T
        return configs['3T']

class OptimizedFMRIPrepProcessor:
    def __init__(self, bids_root: str, output_dir: str, work_dir: str, 
                 fs_license: str, use_docker: bool = True):
        self.bids_root = Path(bids_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.work_dir = Path(work_dir).resolve()
        self.fs_license = Path(fs_license).resolve()
        self.use_docker = use_docker
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.work_dir / 'bids_db'
        logger.info(f"Initializing BIDS layout with database at {db_path}")
        self.layout = BIDSLayout(
            str(self.bids_root),
            database_path=str(db_path),
            reset_database=False
        )
        self.system_info = self._detect_system_resources()
        logger.info(f"System resources: {self.system_info}")
        if self.use_docker:
            self.docker_image = 'nipreps/fmriprep:latest'
            self._check_docker_setup()
    
    def _detect_system_resources(self) -> Dict:
        try:
            cpu_count = psutil.cpu_count(logical=True)
            cpu_physical = psutil.cpu_count(logical=False)
            mem = psutil.virtual_memory()
            total_mem_gb = mem.total / (1024**3)
            available_mem_gb = mem.available / (1024**3)
            gpu_info = self._detect_gpus()
            
            return {
                'cpu_count': cpu_count,
                'cpu_physical': cpu_physical,
                'total_memory_gb': total_mem_gb,
                'available_memory_gb': available_mem_gb,
                'gpus': gpu_info,
                'optimal_threads': min(cpu_physical, 16),  
                'optimal_memory_gb': int(available_mem_gb * 0.8)  
            }
        except Exception as e:
            logger.error(f"Error detecting system resources: {e}")
            return {
                'cpu_count': 8,
                'cpu_physical': 4,
                'total_memory_gb': 32,
                'available_memory_gb': 16,
                'gpus': [],
                'optimal_threads': 8,
                'optimal_memory_gb': 12
            }
    
    def _detect_gpus(self) -> List[Dict]:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    idx, name, memory = line.split(', ')
                    gpus.append({
                        'index': int(idx),
                        'name': name,
                        'memory': memory
                    })
            return gpus
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("No NVIDIA GPUs detected or nvidia-smi not available")
            return []
    
    def _check_docker_setup(self) -> None:
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
            logger.info("Docker is installed")
            result = subprocess.run(
                ['docker', 'images', '-q', self.docker_image],
                capture_output=True, text=True
            )
            
            if not result.stdout.strip():
                logger.info(f"Pulling fMRIPrep Docker image: {self.docker_image}")
                subprocess.run(['docker', 'pull', self.docker_image], check=True)
            if self.system_info['gpus']:
                result = subprocess.run(
                    ['docker', 'run', '--rm', '--gpus', 'all', 'nvidia/cuda:11.0-base', 'nvidia-smi'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    logger.info("NVIDIA Docker runtime is properly configured")
                else:
                    logger.warning("NVIDIA Docker runtime not available, GPU acceleration disabled")
                    self.system_info['gpus'] = []
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker setup error: {e}")
            raise RuntimeError("Docker is not properly installed or configured")
    def build_fmriprep_command(self,
                               subject_id: str,
                               session_id: Optional[str] = None) -> List[str]:
        scanner = ScannerConfig.from_session_id(session_id or '')
        if self.use_docker:
            cmd = [
                'docker', 'run', '--rm',
                '-v', f'{self.bids_root}:/data:ro',
                '-v', f'{self.output_dir}:/out',
                '-v', f'{self.work_dir}:/work',
                '-v', f'{self.fs_license}:/opt/freesurfer/license.txt',
                '-e', 'SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow',
            ]
            if self.system_info['gpus']:
                cmd += ['--gpus', 'all']
            cmd += [
                '-m', f"{self.system_info['optimal_memory_gb']}g",
                '--cpus', str(self.system_info['optimal_threads']),
                self.docker_image,
                '/data', '/out', 'participant'
            ]
        else:
            cmd = ['fmriprep', str(self.bids_root), str(self.output_dir), 'participant']
        cmd += [
            '--participant-label', subject_id,
            '--fs-license-file',
                ('/opt/freesurfer/license.txt' if self.use_docker else str(self.fs_license)),
            '--work-dir', ('/work' if self.use_docker else str(self.work_dir)),
            '--nthreads', str(self.system_info['optimal_threads']),
            '--omp-nthreads', str(max(1, self.system_info['optimal_threads']//2)),
            '--mem-mb', str(int(self.system_info['optimal_memory_gb']*1024)),  # MB
            '--low-mem',
            '--output-spaces'
        ] + scanner.output_spaces + [
            '--fd-spike-threshold', str(scanner.fd_spike_threshold),
            '--dvars-spike-threshold', str(scanner.dvars_spike_threshold),
            '--skull-strip-template', 'OASIS30ANTs',
            '--skull-strip-fixed-seed',
            '--skull-strip-t1w', 'force',
            '--write-graph',
            '--stop-on-first-crash',
            '--notrack',
            '--resource-monitor'
        ]
        cmd += [
            '--bold2anat-init', scanner.bold2anat_init,
            '--bold2anat-dof', str(scanner.bold2anat_dof)
        ]
        if scanner.field_strength == '7T':
            cmd += ['--force', 'bbr']
        if scanner.use_syn_sdc:
            cmd += ['--use-syn-sdc', 'warn']
        cmd += [
            '--skip-bids-validation',
            '--ignore', 'slicetiming',
            '--cifti-output', '91k'
        ]

        return cmd


    def process_subject(self, subject_id: str, session_id: Optional[str] = None,
                       force_rerun: bool = False) -> Tuple[bool, float]:
        if not force_rerun and self._is_processed(subject_id, session_id):
            logger.info(f"Subject {subject_id} session {session_id} already processed, skipping")
            return True, 0.0
        
        start_time = time.time()
        cmd = self.build_fmriprep_command(subject_id, session_id)
        
        logger.info(f"Processing subject {subject_id} session {session_id}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            logger.setLevel(logging.DEBUG)
            logger.debug("Full fMRIPrep command:\n%s", " ".join(cmd))
            for line in process.stdout:
                if line.strip():
                    logger.debug(f"[fMRIPrep] {line.strip()}")
            return_code = process.wait()
            elapsed_time = time.time() - start_time
            
            if return_code == 0:
                logger.info(f"Successfully processed {subject_id} in {elapsed_time/60:.1f} minutes")
                return True, elapsed_time
            else:
                logger.error(f"Failed to process {subject_id} (return code: {return_code})")
                return False, elapsed_time
                
        except Exception as e:
            logger.error(f"Exception processing {subject_id}: {e}")
            return False, time.time() - start_time
    
    def _is_processed(self, subject_id: str, session_id: Optional[str] = None) -> bool:
        fs_dir = self.output_dir / 'freesurfer' / f'sub-{subject_id}'
        if not fs_dir.exists():
            return False
        if session_id:
            html_file = self.output_dir / f'sub-{subject_id}' / f'ses-{session_id}.html'
        else:
            html_file = self.output_dir / f'sub-{subject_id}.html'
        
        return html_file.exists()
    
    def process_dataset(self, subject_list: Optional[List[str]] = None,
                       parallel_jobs: int = 1, force_rerun: bool = False) -> Dict:
        if subject_list is None:
            subject_list = self.layout.get_subjects()
        
        logger.info(f"Processing {len(subject_list)} subjects with {parallel_jobs} parallel jobs")
        processing_queue = []
        for subject in subject_list:
            sessions = self.layout.get_sessions(subject=subject)
            if sessions:
                for session in sessions:
                    processing_queue.append((subject, session))
            else:
                processing_queue.append((subject, None))
        
        logger.info(f"Total processing jobs: {len(processing_queue)}")
        results = {
            'successful': [],
            'failed': [],
            'total_time': 0,
            'average_time': 0
        }
        
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            future_to_job = {
                executor.submit(self.process_subject, subj, sess, force_rerun): (subj, sess)
                for subj, sess in processing_queue
            }
            for future in as_completed(future_to_job):
                subject, session = future_to_job[future]
                try:
                    success, elapsed = future.result()
                    if success:
                        results['successful'].append((subject, session))
                    else:
                        results['failed'].append((subject, session))
                    results['total_time'] += elapsed
                except Exception as e:
                    logger.error(f"Exception for {subject}/{session}: {e}")
                    results['failed'].append((subject, session))
        
        total_jobs = len(results['successful']) + len(results['failed'])
        if total_jobs > 0:
            results['average_time'] = results['total_time'] / total_jobs
        logger.info(f"\nProcessing complete:")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        logger.info(f"  Total time: {results['total_time']/3600:.1f} hours")
        logger.info(f"  Average time per subject: {results['average_time']/60:.1f} minutes")
        
        if results['failed']:
            logger.warning(f"Failed subjects: {results['failed']}")
        
        return results

def create_slurm_script(bids_root: str, output_dir: str, work_dir: str, 
                       fs_license: str, n_subjects: int) -> str:
    """Generate optimized SLURM submission script"""
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=fmriprep-ds005366
#SBATCH --output=logs/fmriprep-%A_%a.out
#SBATCH --error=logs/fmriprep-%A_%a.err
#SBATCH --array=1-{n_subjects}%5  # Run 5 jobs in parallel
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --partition=gpu  # GPU partition

# Load required modules
module load singularity/3.8.0
module load cuda/11.4

# Set up environment
export SINGULARITYENV_TEMPLATEFLOW_HOME=$HOME/.cache/templateflow
export SINGULARITYENV_FS_LICENSE=/opt/freesurfer/license.txt

# Paths
BIDS_ROOT="{bids_root}"
OUTPUT_DIR="{output_dir}"
WORK_DIR="{work_dir}"
FS_LICENSE="{fs_license}"

# Get subject list
subjects=($(ls -d ${{BIDS_ROOT}}/sub-* | xargs -n 1 basename | sort))
SUBJECT=${{subjects[$((SLURM_ARRAY_TASK_ID - 1))]}}
SUBJECT_ID=${{SUBJECT#sub-}}

echo "Processing subject: $SUBJECT_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Create subject-specific work directory
SUBJECT_WORK="${{WORK_DIR}}/sub-${{SUBJECT_ID}}"
mkdir -p "${{SUBJECT_WORK}}"

# Run fMRIPrep with Singularity (GPU-enabled)
singularity run --cleanenv \\
    --bind "${{BIDS_ROOT}}:/data:ro" \\
    --bind "${{OUTPUT_DIR}}:/out" \\
    --bind "${{SUBJECT_WORK}}:/work" \\
    --bind "${{FS_LICENSE}}:/opt/freesurfer/license.txt" \\
    --nv \\
    docker://nipreps/fmriprep:latest \\
    /data /out participant \\
    --participant-label "${{SUBJECT_ID}}" \\
    --work-dir /work \\
    --nthreads ${{SLURM_CPUS_PER_TASK}} \\
    --omp-nthreads 4 \\
    --mem 30gb \\
    --output-spaces MNI152NLin2009cAsym:res-2 T1w fsaverage5 \\
    --longitudinal \\
    --use-syn-sdc \\
    --bold2t1w-dof 9 \\
    --skull-strip-template OASIS30ANTs \\
    --skull-strip-fixed-seed \\
    --write-graph \\
    --resource-monitor \\
    --notrack

# Clean up work directory if successful
if [ $? -eq 0 ]; then
    echo "Processing completed successfully, cleaning work directory"
    rm -rf "${{SUBJECT_WORK}}"
else
    echo "Processing failed, keeping work directory for debugging"
fi
"""
    
    return script_content

def main():
    parser = argparse.ArgumentParser(
        description='Optimized fMRIPrep batch processing for multi-scanner datasets'
    )
    parser.add_argument('bids_root', help='Path to BIDS dataset')
    parser.add_argument('output_dir', help='Path to output directory')
    parser.add_argument('work_dir', help='Path to working directory')
    parser.add_argument('fs_license', help='Path to FreeSurfer license file')
    
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to process')
    parser.add_argument('--parallel', type=int, default=1, 
                       help='Number of parallel jobs (default: 1)')
    parser.add_argument('--no-docker', action='store_true',
                       help='Use native fMRIPrep instead of Docker')
    parser.add_argument('--force-rerun', action='store_true',
                       help='Force rerun even if outputs exist')
    parser.add_argument('--create-slurm-script', action='store_true',
                       help='Create SLURM submission script')
    
    args = parser.parse_args()
    if args.create_slurm_script:
        layout = BIDSLayout(args.bids_root)
        n_subjects = len(layout.get_subjects())
        
        script = create_slurm_script(
            args.bids_root, args.output_dir, 
            args.work_dir, args.fs_license, n_subjects
        )
        
        with open('submit_fmriprep.sh', 'w') as f:
            f.write(script)
        
        print(f"Created SLURM script: submit_fmriprep.sh")
        print(f"Submit with: sbatch submit_fmriprep.sh")
        return
    
    # Initialize processor
    processor = OptimizedFMRIPrepProcessor(
        args.bids_root,
        args.output_dir,
        args.work_dir,
        args.fs_license,
        use_docker=not args.no_docker
    )
    
    # Process dataset
    results = processor.process_dataset(
        subject_list=args.subjects,
        parallel_jobs=args.parallel,
        force_rerun=args.force_rerun
    )
    
    # Save results
    results_file = Path(args.output_dir) / 'processing_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()