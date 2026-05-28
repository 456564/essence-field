"""
算子版本切换工具

用法：
  python scripts/switch_operators.py gray   → 切到灰度版
  python scripts/switch_operators.py rgb    → 切到 RGB 版
  python scripts/switch_operators.py list   → 查看当前版本
"""

import sys, shutil
from pathlib import Path

SRC = Path("src")

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/switch_operators.py [gray|rgb|list]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "gray":
        shutil.copy2(SRC / "operators_gray.py", SRC / "operators.py")
        print("✅ 已切换到 灰度版 (operators_gray.py → operators.py)")
    
    elif cmd == "rgb":
        shutil.copy2(SRC / "operators_rgb.py", SRC / "operators.py")
        print("✅ 已切换到 RGB版 (operators_rgb.py → operators.py)")
    
    elif cmd == "list":
        import filecmp
        if not (SRC / "operators.py").exists():
            print("⚠️  operators.py 不存在")
        elif filecmp.cmp(SRC / "operators.py", SRC / "operators_gray.py"):
            print("📄 当前版本: 灰度版 (operators_gray.py)")
        elif filecmp.cmp(SRC / "operators.py", SRC / "operators_rgb.py"):
            print("📄 当前版本: RGB 版 (operators_rgb.py)")
        else:
            print("⚠️  operators.py 与两个备份都不一致")
    
    else:
        print(f"未知命令: {cmd}")
        print("用法: python scripts/switch_operators.py [gray|rgb|list]")

if __name__ == "__main__":
    main()
