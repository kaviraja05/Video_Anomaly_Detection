"""
Setup script to download or configure I3D pretrained weights.

Run this script to set up the I3D weights needed for raw video feature extraction.
"""

import os
import sys

def check_weights_exist(weights_path: str = "pretrained/rgb_imagenet.pt") -> bool:
    """Check if weights file exists and has reasonable size."""
    if os.path.exists(weights_path):
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        if size_mb > 100:  # I3D weights should be ~400MB
            print(f"✓ Found I3D weights: {weights_path} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"✗ Found file but too small ({size_mb:.1f} MB), may be corrupted")
            return False
    return False


def download_weights():
    """Try to download weights from multiple sources."""
    import urllib.request
    import urllib.error
    
    output_path = "pretrained/rgb_imagenet.pt"
    os.makedirs("pretrained", exist_ok=True)
    
    # Multiple sources to try
    sources = [
        ("Hugging Face", "https://huggingface.co/spaces/gunnit/i3d-kinetics/resolve/main/rgb_imagenet.pt"),
        ("Dropbox", "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_rgb_imagenet.pt?dl=1"),
        ("GitHub Alt", "https://github.com/hassony2/kinetics_i3d_pytorch/releases/download/v0.1.0/rgb_imagenet.pt"),
    ]
    
    print("\nAttempting to download I3D pretrained weights...")
    print("File size: ~400MB, this may take a few minutes...\n")
    
    for name, url in sources:
        print(f"[{name}] Trying: {url}")
        try:
            # Show download progress
            def report_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, block_num * block_size * 100 / total_size)
                    print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
            
            urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
            print()  # New line after progress
            
            if check_weights_exist(output_path):
                return True
                
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
            print(f"  Failed: {e}\n")
            continue
    
    return False


def print_manual_instructions():
    """Print instructions for manual download."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    print("""
The automatic download failed. Please download the I3D weights manually:

OPTION 1 - Original Repository (Recommended):
  1. Go to: https://github.com/piergiaj/pytorch-i3d
  2. Look in the README for pretrained weights download links
  3. Download: rgb_imagenet.pt
  4. Place in: pretrained/rgb_imagenet.pt

OPTION 2 - Alternative GitHub Mirror:
  1. Go to: https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph/tree/master/pretrained
  2. Download the I3D weights file
  3. Rename to rgb_imagenet.pt and place in: pretrained/

OPTION 3 - Use pre-extracted features only (Simplest):
  The demo works perfectly with .npy files from the data/i3d_features folder.
  Just use the "Upload Features" or "Demo Video" tabs instead of raw video upload.
  
  Your test features are already available in: data/i3d_features/test/

OPTION 4 - Contact the original authors:
  The I3D weights were originally from the paper:
  "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
  Check Google Scholar or the authors' websites for download links.

After downloading, run this script again to verify:
  python feature_extraction/setup_weights.py
""")
    print("=" * 70)


def main():
    print("=" * 70)
    print("I3D Weights Setup for Video Anomaly Detection")
    print("=" * 70)
    
    weights_path = "pretrained/rgb_imagenet.pt"
    
    # Check if weights already exist
    if check_weights_exist(weights_path):
        print("\n✓ I3D weights are ready!")
        print("  You can now use raw video upload in the Streamlit demo.")
        return 0
    
    print(f"\n✗ Weights not found at: {os.path.abspath(weights_path)}")
    
    # Ask user if they want to attempt download
    try:
        response = input("\nAttempt automatic download? [Y/n]: ").strip().lower()
        if response in ['', 'y', 'yes']:
            if download_weights():
                print("\n✓ Setup complete! I3D weights are ready.")
                return 0
            else:
                print_manual_instructions()
                return 1
        else:
            print_manual_instructions()
            return 1
    except (KeyboardInterrupt, EOFError):
        print("\n\nCancelled.")
        print_manual_instructions()
        return 1


if __name__ == "__main__":
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    sys.exit(main())
