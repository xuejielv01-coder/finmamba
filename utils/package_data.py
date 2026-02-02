
import zipfile
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger

logger = get_logger("DataPackager")

def package_data():
    """Package essential data for Colab training"""
    source_dirs = [
        project_root / "data" / "processed",
        project_root / "data" / "fast_cache",
        project_root / "data" / "stats"
    ]
    
    output_filename = project_root / "deepalpha_data_pack.zip"
    
    logger.info(f"Starting data packaging to {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for source_dir in source_dirs:
            if not source_dir.exists():
                logger.warning(f"Directory not found, skipping: {source_dir}")
                continue
                
            logger.info(f"Adding directory: {source_dir.name}")
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(project_root)
                    zf.write(file_path, arcname)
                    
    logger.info(f"Packaging complete. File size: {output_filename.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\n[SUCCESS] Data package created at: {output_filename}")
    print("Please upload this file to your Google Drive root directory.")

if __name__ == "__main__":
    package_data()
