# -*- mode: python -*-

block_cipher = None

data_paths = [('docs\sample_data\slide01_section1_area08.tif', 'sample_data'),
              (r'venv\Lib\site-packages\PyQt5\Qt\plugins\platforms\qminimal.dll', 'platforms'),
              (r'venv\Lib\site-packages\PyQt5\Qt\plugins\platforms\qoffscreen.dll', 'platforms'),
              (r'venv\Lib\site-packages\PyQt5\Qt\plugins\platforms\qwindows.dll', 'platforms'),
              ]

a = Analysis(['axondegen\\degengui.py'],
             pathex=[
                     r'.',
                     r'env',
                     ],
             binaries=[],
             datas=data_paths,
             hiddenimports=['scipy._lib.messagestream', 'pyqtgraph'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='axondegen',
          debug=True,
          strip=False,
          upx=True,
          console=True )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='axondegen')
