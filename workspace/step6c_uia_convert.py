"""
Step 6c: uiautomation 自动化 CAD Assistant .x_t → STL 转换
"""
import os, sys, time, subprocess
from pathlib import Path
import uiautomation as auto

CAD_EXE = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\workspace\CAD_Assistant\CADAssistant.exe")
DATA_DIR = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\DecPelton2024\DecPelton2024\DEC PELTON 2024-150MW级冲击式国产化转轮研发共享数据包")
OUTPUT_DIR = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\workspace\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    (DATA_DIR / "1-转轮水力开发数据" / "1.1-金窝模型转轮D1276P21_.x_t",
     OUTPUT_DIR / "dec_model_runner.stl", "D1276P21"),
    (DATA_DIR / "2-真机转轮三维实体数据" / "2.1-150MW级冲击式国产化真机转轮三维实体数据.x_t",
     OUTPUT_DIR / "dec_full_runner.stl", "D1247R2"),
]

auto.SetGlobalSearchTimeout(3)


def kill_cad():
    os.system('taskkill /f /im CADAssistant.exe 2>nul >nul')
    time.sleep(1)


def find_cad_window(timeout=30):
    """找 CAD Assistant 主窗口"""
    start = time.time()
    while time.time() - start < timeout:
        # 搜索所有顶层窗口
        root = auto.GetRootControl()
        for w in root.GetChildren():
            if 'CAD Assistant' in (w.Name or ''):
                return w
        time.sleep(1)
    return None


def convert_one(xt_path, stl_path, label):
    """UIA 自动化转换"""
    print(f"\n{'='*50}")
    print(f"🔄 {label}")
    
    if stl_path.exists() and stl_path.stat().st_size > 500000:
        print(f"   ✅ 已存在, 跳过")
        return True
    
    kill_cad()
    
    # 启动 (不加载文件, 先启动空程序)
    print(f"   启动 CAD Assistant...")
    subprocess.Popen([str(CAD_EXE)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    
    window = find_cad_window()
    if not window:
        print(f"   ❌ 窗口未出现")
        kill_cad()
        return False
    
    print(f"   窗口: {window.Name}")
    window.SetFocus()
    time.sleep(1)
    
    # === 方法: 用 Alt+F → O (Open) 然后导入文件 ===
    window.SendKeys('{Alt}f')  # Alt+F = File menu
    time.sleep(0.5)
    window.SendKeys('o')  # Open
    time.sleep(2)
    
    # 在打开文件对话框中输入路径
    xt_str = str(xt_path.resolve())
    window.SendKeys(xt_str)
    time.sleep(0.5)
    window.SendKeys('{Enter}')
    time.sleep(15)  # 等待47MB文件加载
    
    # 重新找窗口 (标题会变)
    window = find_cad_window()
    if window:
        print(f"   加载后窗口: {window.Name[:60]}...")
        window.SetFocus()
        time.sleep(1)
    
    # === 导出: File → Export 或 Ctrl+Shift+E ===
    print(f"   导出 STL...")
    window.SendKeys('{Alt}f')  # File menu
    time.sleep(0.5)
    window.SendKeys('e')  # Export (或者 Export to...)
    time.sleep(2)
    
    # 尝试找导出对话框
    # 输入文件路径
    stl_str = str(stl_path.resolve())
    window.SendKeys(stl_str)
    time.sleep(0.5)
    window.SendKeys('{Enter}')
    time.sleep(5)
    
    # 可能弹出格式选择对话框, 再按一次 Enter
    window.SendKeys('{Enter}')
    time.sleep(5)
    
    # 检查是否生成
    if stl_path.exists() and stl_path.stat().st_size > 500000:
        size_mb = stl_path.stat().st_size / 1024 / 1024
        print(f"   ✅ 成功! ({size_mb:.1f} MB)")
        window.SendKeys('{Alt}{F4}')
        time.sleep(1)
        kill_cad()
        return True
    
    # === 备用方法: 尝试 Ctrl+Shift+S (Save As) ===
    print(f"   方法1失败, 尝试 Save As...")
    window.SetFocus()
    time.sleep(0.5)
    window.SendKeys('{Ctrl}{Shift}s')
    time.sleep(2)
    
    window.SendKeys(stl_str)
    time.sleep(0.5)
    window.SendKeys('{Enter}')
    time.sleep(5)
    window.SendKeys('{Enter}')
    time.sleep(5)
    
    # 关闭
    window.SendKeys('{Alt}{F4}')
    time.sleep(1)
    kill_cad()
    
    if stl_path.exists() and stl_path.stat().st_size > 500000:
        size_mb = stl_path.stat().st_size / 1024 / 1024
        print(f"   ✅ 成功! ({size_mb:.1f} MB)")
        return True
    elif stl_path.exists():
        print(f"   ⚠️ 文件太小 ({stl_path.stat().st_size} bytes)")
        stl_path.unlink()
    
    print(f"   ❌ 失败")
    return False


def main():
    print("="*60)
    print("🔧 UIA 自动化 .x_t → STL")
    print("="*60)
    
    kill_cad()
    
    for xt_path, stl_path, label in FILES:
        if not xt_path.exists():
            print(f"❌ {label}: 文件不存在")
            continue
        try:
            convert_one(xt_path, stl_path, label)
        except Exception as e:
            print(f"   ❌ 异常: {e}")
            kill_cad()
    
    print(f"\n📋 结果:")
    for _, stl_path, label in FILES:
        if stl_path.exists():
            size_mb = stl_path.stat().st_size / 1024 / 1024
            status = "✅" if size_mb > 0.5 else "⚠️"
            print(f"   {status} {label}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {label}: 未生成")
    
    kill_cad()


if __name__ == "__main__":
    main()
