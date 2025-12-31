# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

# 收集 mmcv 的 C++ 扩展 (.pyd / .dll)
mmcv_binaries = collect_dynamic_libs('mmcv')

# 防止动态 import 丢失
hiddenimports = (
    collect_submodules('mmcv') +
    collect_submodules('mmpose')
)

a = Analysis(
    ['app_gui.py'],
    pathex=['.'],
    binaries=mmcv_binaries,
    datas=[
        ('assets', 'assets'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PoseLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    icon='PoseLab.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PoseLab',
)
