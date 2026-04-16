#!/usr/bin/env python3
"""Install the Gollek Jupyter kernel."""

import sys, json, shutil, subprocess, argparse
from pathlib import Path

JAR_NAME = "gollek-kernel.jar"

def main():
    parser = argparse.ArgumentParser(description="Install Gollek Jupyter Kernel")
    parser.add_argument("--user", action="store_true")
    parser.add_argument("--sys-prefix", action="store_true")
    parser.add_argument("--prefix")
    parser.add_argument("--kernel-name", default="gollek")
    parser.add_argument("--display-name", default="Gollek (Java 25 + ML)")
    args = parser.parse_args()

    jar_src = Path(__file__).parent / "target" / JAR_NAME
    if not jar_src.exists():
        print(f"❌ JAR not found. Run: mvn clean package")
        sys.exit(1)

    if args.sys_prefix:
        base = Path(sys.prefix) / "share" / "jupyter" / "kernels"
    elif args.prefix:
        base = Path(args.prefix) / "share" / "jupyter" / "kernels"
    elif args.user:
        base = Path.home() / ".local" / "share" / "jupyter" / "kernels"
    else:
        result = subprocess.run(["jupyter", "--data-dir"], capture_output=True, text=True)
        base = Path(result.stdout.strip()) / "kernels"

    install_dir = base / args.kernel_name
    install_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(jar_src, install_dir / JAR_NAME)

    tpl = Path(__file__).parent / "src" / "main" / "resources" / "kernel.json"
    config = json.loads(tpl.read_text())
    config["display_name"] = args.display_name
    config["argv"] = [a.replace("{install_dir}", str(install_dir)) for a in config["argv"]]
    (install_dir / "kernel.json").write_text(json.dumps(config, indent=2))

    subprocess.run(
        ["jupyter", "kernelspec", "install", str(install_dir),
         "--name", args.kernel_name, "--replace"] +
        (["--user"] if args.user else []),
        check=True, capture_output=True
    )
    print(f"✅ Gollek kernel installed: {install_dir}")
    print(f"   Verify: jupyter kernelspec list")

if __name__ == "__main__":
    main()
