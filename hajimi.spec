# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# 收集 ultralytics 的数据和依赖
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# 收集 sentence_transformers 的数据
tmp_ret = collect_all('sentence_transformers')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# 添加项目自己的数据文件
# 注意：Vosk 模型和 YOLO 模型因为体积大，建议不打包进 exe，而是放在同级目录
# 这里我们假设用户会把模型放在 exe 同级目录，所以只打包代码需要的资源
datas += [('hajimi.png', '.')]

# 添加必要的 hidden imports
hiddenimports += ['sklearn.utils._typedefs', 'sklearn.neighbors._partition_nodes']

block_cipher = None

a = Analysis(
    ['hajimi.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HajimiAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # 设为 True 可以看到 print 的错误信息，发布时可改为 False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='hajimi.png' # 如果有 ico 文件最好用 ico
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HajimiAI',
)
