"""
Step 6: pywinauto 自动化 CAD Assistant 转换 .x_t → STL
"""
import os, sys, time, subprocess
from pathlib import Path

CAD_EXE = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\workspace\CAD_Assistant\CADAssistant.exe")
DATA_DIR = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\DecPelton2024\DecPelton2024\DEC PELTON 2024-150MW级冲击式国产化转轮研发共享数据包")
OUTPUT_DIR = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\workspace\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    (DATA_DIR / "1-转轮水力开发数据" / "1.1-金窝模型转轮D1276P21_.x_t",
     OUTPUT_DIR / "dec_model_runner.stl",
     "模型转轮 D1276P21"),
    (DATA_DIR / "2-真机转轮三维实体数据" / "2.1-150MW级冲击式国产化真机转轮三维实体数据.x_t",
     OUTPUT_DIR / "dec_full_runner.stl",
     "真机转轮 D1247R2"),
]


def kill_existing():
    """杀掉所有已有的 CAD Assistant 进程"""
    os.system('taskkill /f /im CADAssistant.exe 2>nul >nul')
    time.sleep(1)


def convert_one(xt_path, stl_path, label):
    """转换单个文件"""
    from pywinauto import Application
    from pywinauto.keyboard import send_keys
    
    print(f"\n{'='*50}")
    print(f"🔄 {label}")
    print(f"   输入: {xt_path.name}")
    
    if stl_path.exists():
        size_mb = stl_path.stat().st_size / 1024 / 1024
        print(f"   ✅ 已存在 ({size_mb:.1f} MB), 跳过")
        return True
    
    kill_existing()
    
    # 启动 CAD Assistant
    xt_abs = str(xt_path.resolve())
    print(f"   启动 CAD Assistant...")
    
    app = Application(backend="uia").start(f'"{CAD_EXE}" "{xt_abs}"')
    time.sleep(8)  # 等待加载大文件
    
    try:
        # 连接主窗口
        main_window = app.window()
        main_window.wait('visible', timeout=30)
        main_window.set_focus()
        print(f"   窗口已就绪")
        
        # 导出: Ctrl+Shift+E 或 File → Export
        # CAD Assistant 菜单: File → Export
        send_keys('%f')  # Alt+F (File menu)
        time.sleep(0.5)
        
        # 查看是否有 Export 菜单项
        # 尝试直接 Ctrl+E
        send_keys('^+e')  # Ctrl+Shift+E (common Export shortcut)
        time.sleep(2)
        
        # 查找导出对话框
        try:
            export_dlg = app.window(title_re=".*Export.*|.*Save.*|.*导出.*|.*另存为.*", timeout=5)
            export_dlg_exists = export_dlg.exists()
        except:
            export_dlg_exists = False
        
        if not export_dlg_exists:
            # 尝试另一种方式
            send_keys('%f')  # Alt+F
            time.sleep(0.5)
            send_keys('e')   # E for Export
            time.sleep(2)
        
        # 在文件对话框中输入路径
        stl_abs = str(stl_path.resolve())
        try:
            # 尝试用 Ctrl+L 选中地址栏, 或直接输入
            send_keys('^l')  # 地址栏
            time.sleep(0.3)
        except:
            pass
        
        # 输入完整路径
        send_keys(stl_abs)
        time.sleep(0.5)
        send_keys('{ENTER}')
        time.sleep(3)
        
        # 处理可能的覆盖确认
        try:
            confirm = app.window(title_re=".*确认.*|.*覆盖.*|.*overwrite.*|.*replace.*", timeout=2)
            if confirm.exists():
                send_keys('{ENTER}')  # Yes/OK
                time.sleep(2)
        except:
            pass
        
        # 等待导出完成
        time.sleep(5)
        
        # 关闭 CAD Assistant
        send_keys('%{F4}')  # Alt+F4
        time.sleep(1)
        
        # 强制关闭
        kill_existing()
        
    except Exception as e:
        print(f"   ⚠️ 异常: {e}")
        try:
            app.kill()
        except:
            pass
        kill_existing()
    
    # 验证
    if stl_path.exists():
        size_mb = stl_path.stat().st_size / 1024 / 1024
        if size_mb > 0.1:  # 至少100KB
            print(f"   ✅ 成功! ({size_mb:.1f} MB)")
            return True
        else:
            print(f"   ⚠️ 文件太小 ({size_mb:.2f} MB), 可能为空")
            stl_path.unlink()
    else:
        print(f"   ❌ 未生成文件")
    
    return False


def convert_manual_method():
    """手动方法：逐键操作"""
    from pywinauto import Application
    from pywinauto.keyboard import send_keys
    
    for xt_path, stl_path, label in FILES:
        if stl_path.exists():
            continue
            
        kill_existing()
        print(f"\n🔄 {label} (手动方法)")
        
        app = Application(backend="uia").start(f'"{CAD_EXE}"')
        time.sleep(5)
        
        main_window = app.window()
        main_window.wait('visible', timeout=30)
        main_window.set_focus()
        
        # File → Import
        time.sleep(1)
        send_keys('^o')  # Ctrl+O = Open
        time.sleep(2)
        
        # 输入文件路径
        send_keys(str(xt_path.resolve()))
        time.sleep(1)
        send_keys('{ENTER}')
        time.sleep(10)  # 等待大文件加载
        
        # File → Export
        send_keys('^+s')  # Ctrl+Shift+S = Save As/Export
        time.sleep(2)
        
        # 输入导出路径
        send_keys(str(stl_path.resolve()))
        time.sleep(0.5)
        send_keys('{TAB}')  # 跳到格式选择
        time.sleep(0.3)
        send_keys('s')  # STL
        time.sleep(0.3)
        send_keys('{ENTER}')
        time.sleep(5)
        
        # 关闭
        send_keys('%{F4}')
        time.sleep(1)
        kill_existing()
        
        if stl_path.exists():
            size_mb = stl_path.stat().st_size / 1024 / 1024
            print(f"   ✅ {size_mb:.1f} MB")


def main():
    print("="*60)
    print("🔧 自动化 .x_t → STL 转换")
    print("="*60)
    
    kill_existing()
    
    # 方法1: 直接尝试
    for xt_path, stl_path, label in FILES:
        if not xt_path.exists():
            print(f"❌ 文件不存在: {xt_path}")
            continue
        convert_one(xt_path, stl_path, label)
    
    # 如果方法1失败, 尝试方法2
    for xt_path, stl_path, label in FILES:
        if not stl_path.exists() and xt_path.exists():
            convert_manual_method()
            break
    
    # 最终报告
    print(f"\n{'='*50}")
    print(f"📋 转换结果:")
    all_ok = True
    for _, stl_path, label in FILES:
        if stl_path.exists():
            size_mb = stl_path.stat().st_size / 1024 / 1024
            print(f"   ✅ {label}: {stl_path.name} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {label}: 失败")
            all_ok = False
    
    if all_ok:
        print(f"\n✅ 全部转换完成, 可以运行三级验证!")
    
    kill_existing()


if __name__ == "__main__":
    main()
