# -*- coding: utf-8 -*-
"""
Script to package AVIR_KIE_Submission_Package into a clean zip distribution.
Excludes unnecessary build artifacts, pycache, and temporary files.
"""

import os
import sys
import zipfile
import shutil

sys.stdout.reconfigure(encoding='utf-8')

SOURCE_DIR = r"c:\Users\Admin\OneDrive\DoAn\AVIR_KIE_Submission_Package"
OUTPUT_ZIP = r"c:\Users\Admin\OneDrive\DoAn\AVIR_KIE_Capstone_SourceCode_Final.zip"
OUTPUT_LATEST_ZIP = r"c:\Users\Admin\OneDrive\DoAn\AVIR_KIE_Capstone_SourceCode_Latest.zip"
ARTIFACT_DIR = r"C:\Users\Admin\.gemini\antigravity\brain\37cbc4ef-0254-4bd1-9c06-47e50fbf99ff"

EXCLUDE_DIRS = {
    "__pycache__",
    "node_modules",
    ".next",
    ".git",
    "temp_uploads",
    ".turbo",
    ".cache"
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".DS_Store"
}

def create_zip():
    print(f"📦 Packaging {SOURCE_DIR} into {OUTPUT_ZIP}...")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(SOURCE_DIR):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in EXCLUDE_EXTS or file.startswith("."):
                    if file not in [".env", ".gitignore"]:
                        continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, os.path.dirname(SOURCE_DIR))
                
                zipf.write(file_path, rel_path)
                file_count += 1
                total_uncompressed_bytes += os.path.getsize(file_path)

    zip_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"✅ Successfully created {OUTPUT_ZIP}")
    print(f"   Total files: {file_count}")
    print(f"   Uncompressed size: {total_uncompressed_bytes / (1024 * 1024):.2f} MB")
    print(f"   Compressed size: {zip_size_mb:.2f} MB")

    # Copy to latest name and to artifacts directory
    shutil.copyfile(OUTPUT_ZIP, OUTPUT_LATEST_ZIP)
    print(f"✅ Created copy at {OUTPUT_LATEST_ZIP}")

    artifact_dest = os.path.join(ARTIFACT_DIR, "AVIR_KIE_Capstone_SourceCode_Final.zip")
    shutil.copyfile(OUTPUT_ZIP, artifact_dest)
    print(f"✅ Copied to artifact location {artifact_dest}")

    artifact_dest_latest = os.path.join(ARTIFACT_DIR, "AVIR_KIE_Capstone_SourceCode_Latest.zip")
    shutil.copyfile(OUTPUT_ZIP, artifact_dest_latest)
    print(f"✅ Copied to artifact location {artifact_dest_latest}")

if __name__ == "__main__":
    create_zip()
