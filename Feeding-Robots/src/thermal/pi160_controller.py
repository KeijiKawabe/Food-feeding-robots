import ctypes
import time
import numpy as np
import cv2
import os



class PI160Controller:
    """libirimager DLLを直接呼び出してPI160を制御"""

    def __init__(self):
        self.lib = None
        self.handle = None
        self.width = 160
        self.height = 120
        self._load_dll()

        if self.lib:
            self._init_camera()

    def _load_dll(self):
        """DLLをロード"""
        dll_dir = r"C:\Program Files (x86)\Optris GmbH\IrDirectSDK\bin\x64"
        dll_path = os.path.join(dll_dir, "libirimager.dll")
        ctypes.windll.kernel32.SetErrorMode(0x0001)  # SEM_FAILCRITICALERRORS

        # DLLの依存パスを登録
        try:
            os.add_dll_directory(dll_dir)
        except:
            pass

        if os.path.exists(dll_path):
            try:
                self.lib = ctypes.CDLL(dll_path)
                print(f"✓ DLLロード成功: {dll_path}")
            except Exception as e:
                print(f"✗ DLLロード失敗: {e}")
        else:
            print(f"✗ DLLが見つかりません: {dll_path}")
            print("  IrDirectSDKがインストールされているか確認してください")

    def _init_camera(self):
        if not self.lib:
            return False

        # --- 関数定義をDirect Bindingヘッダに従って登録 ---
        self.lib.evo_irimager_usb_init.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        self.lib.evo_irimager_usb_init.restype = ctypes.c_int

        self.lib.evo_irimager_terminate.argtypes = []
        self.lib.evo_irimager_terminate.restype = ctypes.c_int

        self.lib.evo_irimager_get_thermal_image_size.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.evo_irimager_get_thermal_image_size.restype = ctypes.c_int

        self.lib.evo_irimager_get_palette_image_size.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.evo_irimager_get_palette_image_size.restype = ctypes.c_int

        self.lib.evo_irimager_get_thermal_image.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ushort)
        ]
        self.lib.evo_irimager_get_thermal_image.restype = ctypes.c_int

        self.lib.evo_irimager_get_palette_image.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ubyte)
        ]
        self.lib.evo_irimager_get_palette_image.restype = ctypes.c_int

        # --- 初期化処理 ---
        config_path = r"C:\Program Files (x86)\Optris GmbH\IrDirectSDK\examples\matlab\generic.xml"
        formats_path = r"C:\Program Files (x86)\Optris GmbH\IrDirectSDK\examples\matlab"

        print(f"  設定ファイル: {config_path}")
        print(f"  Formats.def: {formats_path}")

        result = self.lib.evo_irimager_usb_init(
            config_path.encode('utf-8'),
            formats_path.encode('utf-8'),
            None
        )

        if result == 0:
            print("✓ PI160カメラ初期化成功")
            self.handle = True
            self._get_image_size()
            print(" カメラのウォームアップ待機中...")
            time.sleep(1.0) # 1秒ほど待機してストリームが安定するのを待つ
            return True
        else:
            print(f"✗ 初期化失敗 (コード: {result})")
            return False


    def _get_dummy_frame(self):
        """ダミーフレーム取得（バッファクリア用）"""
        try:
            # ★修正: 関数シグネチャを変更
            get_palette = self.lib.evo_irimager_get_palette_image
            get_palette.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
            get_palette.restype = None  # ★重要: c_intからNoneに変更
            
            size = self.width * self.height * 3
            buffer = (ctypes.c_ubyte * size)()
            get_palette(buffer)
        except:
            pass

    def get_thermal_data(self):
        if not self.lib or not self.handle:
            return None

        width = ctypes.c_int(self.width)
        height = ctypes.c_int(self.height)
        buffer = (ctypes.c_ushort * (self.width * self.height))()

        ret = self.lib.evo_irimager_get_thermal_image(
            ctypes.byref(width), ctypes.byref(height), buffer
        )

        if ret != 0:
            print(f"✗ 温度データ取得失敗 (コード: {ret})")
            return None
        
        data_raw = np.ctypeslib.as_array(buffer).reshape((height.value, width.value))
        temperature = data_raw.astype(np.float32)*0.1 -100.0  # 生データを温度に変換
        # ✅ [テストA2] 上下反転（PIX Connectと向きを合わせる）
       #temperature = np.flipud(temperature)
        print(f"✓ 温度データ取得成功: {temperature.min():.1f}~{temperature.max():.1f}℃")
        # ✅ [テストA3] デバッグ出力
     #  print(f"[DEBUG] Temperature shape: {temperature.shape}")
     #  print(f"[DEBUG] Min={temperature.min():.2f}, Max={temperature.max():.2f}, Mean={temperature.mean():.2f}")

        # --- 🔼 ここまで修正 ---
        return temperature

    def get_palette_image(self):
        if not self.lib or not self.handle:
            return None

        width = ctypes.c_int(self.width)
        height = ctypes.c_int(self.height)
        buffer = (ctypes.c_ubyte * (self.width * self.height * 3))()

        ret = self.lib.evo_irimager_get_palette_image(
            ctypes.byref(width), ctypes.byref(height), buffer
        )

        if ret != 0:
            print(f"✗ パレット画像取得失敗 (コード: {ret})")
            return None

        img = np.ctypeslib.as_array(buffer).reshape((height.value, width.value, 3))
        img_bgr = cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR)
        print(f"✓ パレット画像取得成功: {img_bgr.shape}")
        return img_bgr

    def capture_frame(self):
        """
        ThermalGPTが期待する形式で 1 フレーム（カラー画像 + 温度データ）を返す。
        """
        palette = self.get_palette_image()     # BGR可視画像
        raw     = self.get_thermal_data()      # 温度マップ

        if palette is None or raw is None:
            print("✗ capture_frame：フレーム取得失敗")
            return None, None

        return palette, raw


    def disconnect(self):
        """カメラ切断"""
        if self.lib and self.handle:
            try:
                terminate = self.lib.evo_irimager_terminate
                terminate.restype = None
                terminate()
                print("カメラ切断")
            except:
                pass
    def _get_image_size(self):
        """画像サイズ取得"""
        try:
            get_size = self.lib.evo_irimager_get_thermal_image_size
            get_size.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
            get_size.restype = ctypes.c_int

            w, h = ctypes.c_int(), ctypes.c_int()
            if get_size(ctypes.byref(w), ctypes.byref(h)) == 0:
                self.width, self.height = w.value, h.value
                print(f"  画像サイズ: {self.width}x{self.height}")
            else:
                print("✗ 画像サイズ取得失敗")
        except Exception as e:
            print(f"✗ _get_image_sizeエラー: {e}")
