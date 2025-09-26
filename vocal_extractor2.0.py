#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版人声提取应用程序
专门针对舞台现场录音优化，解决垫音残留问题
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter
from difflib import SequenceMatcher
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QFileDialog, QProgressBar, QSlider, QVBoxLayout,
                             QHBoxLayout, QWidget, QMessageBox, QGroupBox, QLineEdit,
                             QComboBox, QDialog, QSpinBox, QCheckBox, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QFont


class SettingsDialog(QDialog):
    """设置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings('VocalExtractor', 'EnhancedVersion')
        self.initUI()
        self.load_settings()
    
    def initUI(self):
        self.setWindowTitle("设置")
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout()
        
        # 音频处理参数组
        audio_group = QGroupBox("音频处理参数")
        audio_layout = QVBoxLayout()
        
        # FFT大小
        fft_layout = QHBoxLayout()
        fft_layout.addWidget(QLabel("FFT大小:"))
        self.fft_spin = QSpinBox()
        self.fft_spin.setMinimum(1024)
        self.fft_spin.setMaximum(8192)
        self.fft_spin.setValue(2048)
        self.fft_spin.setSingleStep(1024)
        self.fft_spin.setToolTip("更大的FFT提供更好的频率分辨率，但处理速度更慢")
        fft_layout.addWidget(self.fft_spin)
        fft_layout.addStretch()
        audio_layout.addLayout(fft_layout)
        
        # 跳跃长度
        hop_layout = QHBoxLayout()
        hop_layout.addWidget(QLabel("跳跃长度:"))
        self.hop_spin = QSpinBox()
        self.hop_spin.setMinimum(128)
        self.hop_spin.setMaximum(2048)
        self.hop_spin.setValue(512)
        self.hop_spin.setSingleStep(128)
        self.hop_spin.setToolTip("较小的跳跃长度提供更好的时间分辨率")
        hop_layout.addWidget(self.hop_spin)
        hop_layout.addStretch()
        audio_layout.addLayout(hop_layout)
        
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        # 算法参数组
        algo_group = QGroupBox("算法参数")
        algo_layout = QVBoxLayout()
        
        # 频谱减法参数
        spectral_layout = QHBoxLayout()
        spectral_layout.addWidget(QLabel("频谱减法系数:"))
        self.alpha_spin = QSpinBox()
        self.alpha_spin.setMinimum(100)
        self.alpha_spin.setMaximum(300)
        self.alpha_spin.setValue(200)
        self.alpha_spin.setSuffix("%")
        self.alpha_spin.setToolTip("控制频谱减法的强度，值越大去除越彻底")
        spectral_layout.addWidget(self.alpha_spin)
        spectral_layout.addStretch()
        algo_layout.addLayout(spectral_layout)
        
        # 最小保留比例
        beta_layout = QHBoxLayout()
        beta_layout.addWidget(QLabel("最小保留比例:"))
        self.beta_spin = QSpinBox()
        self.beta_spin.setMinimum(1)
        self.beta_spin.setMaximum(20)
        self.beta_spin.setValue(5)
        self.beta_spin.setSuffix("%")
        self.beta_spin.setToolTip("防止过度减法造成失真的保护参数")
        beta_layout.addWidget(self.beta_spin)
        beta_layout.addStretch()
        algo_layout.addLayout(beta_layout)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)
        
        # 处理选项组
        option_group = QGroupBox("处理选项")
        option_layout = QVBoxLayout()
        
        self.harmonic_cb = QCheckBox("启用谐波分离预处理")
        self.harmonic_cb.setChecked(True)
        self.harmonic_cb.setToolTip("分离谐波和冲击成分，有助于保留人声特征")
        option_layout.addWidget(self.harmonic_cb)
        
        self.temporal_cb = QCheckBox("启用时域平滑")
        self.temporal_cb.setChecked(True)
        self.temporal_cb.setToolTip("平滑时域变化，减少瞬态噪声")
        option_layout.addWidget(self.temporal_cb)
        
        self.vad_cb = QCheckBox("启用语音活动检测")
        self.vad_cb.setChecked(True)
        self.vad_cb.setToolTip("识别人声片段，抑制纯乐器段落")
        option_layout.addWidget(self.vad_cb)
        
        self.denoise_cb = QCheckBox("启用后处理降噪")
        self.denoise_cb.setChecked(True)
        self.denoise_cb.setToolTip("去除高频噪声和残留")
        option_layout.addWidget(self.denoise_cb)
        
        option_group.setLayout(option_layout)
        layout.addWidget(option_group)
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def load_settings(self):
        """加载设置"""
        self.fft_spin.setValue(self.settings.value('fft_size', 2048, type=int))
        self.hop_spin.setValue(self.settings.value('hop_length', 512, type=int))
        self.alpha_spin.setValue(self.settings.value('alpha', 200, type=int))
        self.beta_spin.setValue(self.settings.value('beta', 5, type=int))
        self.harmonic_cb.setChecked(self.settings.value('enable_harmonic', True, type=bool))
        self.temporal_cb.setChecked(self.settings.value('enable_temporal', True, type=bool))
        self.vad_cb.setChecked(self.settings.value('enable_vad', True, type=bool))
        self.denoise_cb.setChecked(self.settings.value('enable_denoising', True, type=bool))
    
    def save_settings(self):
        """保存设置"""
        self.settings.setValue('fft_size', self.fft_spin.value())
        self.settings.setValue('hop_length', self.hop_spin.value())
        self.settings.setValue('alpha', self.alpha_spin.value())
        self.settings.setValue('beta', self.beta_spin.value())
        self.settings.setValue('enable_harmonic', self.harmonic_cb.isChecked())
        self.settings.setValue('enable_temporal', self.temporal_cb.isChecked())
        self.settings.setValue('enable_vad', self.vad_cb.isChecked())
        self.settings.setValue('enable_denoising', self.denoise_cb.isChecked())
    
    def restore_defaults(self):
        """恢复默认设置"""
        self.fft_spin.setValue(2048)
        self.hop_spin.setValue(512)
        self.alpha_spin.setValue(200)
        self.beta_spin.setValue(5)
        self.harmonic_cb.setChecked(True)
        self.temporal_cb.setChecked(True)
        self.vad_cb.setChecked(True)
        self.denoise_cb.setChecked(True)
    
    def accept(self):
        self.save_settings()
        super().accept()


class EnhancedProcessingThread(QThread):
    """增强版音频处理线程"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    processing_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, song_path, accompaniment_path, output_path, params):
        super().__init__()
        self.song_path = song_path
        self.accompaniment_path = accompaniment_path
        self.output_path = output_path
        self.params = params
        self.is_running = True

    def spectral_subtraction_enhanced(self, song_stft, acc_stft, alpha=2.0, beta=0.05):
        """增强版频谱减法，减少垫音残留"""
        song_mag = np.abs(song_stft)
        song_phase = np.angle(song_stft)
        acc_mag = np.abs(acc_stft)
        
        # 自适应噪声估计
        noise_mag = acc_mag * alpha
        
        # 计算频谱减法
        clean_mag = song_mag - noise_mag
        
        # 过度减法保护，防止产生音乐噪声
        mask = clean_mag / (song_mag + 1e-10)
        mask = np.maximum(mask, beta)  # 设置最小保留比例
        
        # 平滑掩码，减少突变
        mask = median_filter(mask, size=(1, 3))
        
        return song_mag * mask * np.exp(1j * song_phase)

    def adaptive_wiener_filter(self, song_stft, acc_stft, vocal_estimate=None, strength=1.0):
        """自适应维纳滤波器
        Args:
            strength: 滤波强度参数，范围0.0~2.0，1.0为正常强度
        """
        song_mag = np.abs(song_stft)
        acc_mag = np.abs(acc_stft)
        
        if vocal_estimate is not None:
            vocal_mag = np.abs(vocal_estimate)
        else:
            vocal_mag = song_mag - acc_mag
            vocal_mag = np.maximum(vocal_mag, 0.1 * song_mag)
        
        # 计算信噪比
        snr = vocal_mag**2 / (acc_mag**2 + 1e-10)
        
        # 自适应维纳滤波，应用强度参数
        wiener_gain = (snr / (1 + snr)) ** strength
        
        # 频率相关的增益调整（人声频段增强）
        freqs = np.linspace(0, 22050, song_stft.shape[0])
        vocal_boost = np.ones_like(freqs)
        vocal_range = (300 <= freqs) & (freqs <= 3400)  # 人声主要频段
        vocal_boost[vocal_range] *= 1.5
        
        wiener_gain = wiener_gain * vocal_boost.reshape(-1, 1)
        
        return song_stft * wiener_gain

    def harmonic_percussive_separation(self, audio, sr):
        """谐波-冲击分离，保留人声谐波特征"""
        stft = librosa.stft(audio)
        harmonic, percussive = librosa.decompose.hpss(stft, margin=(3.0, 5.0))
        return librosa.istft(harmonic)

    def temporal_smoothing(self, stft_matrix, window_size=5):
        """时域平滑，减少瞬态噪声"""
        smoothed = np.copy(stft_matrix)
        for i in range(stft_matrix.shape[0]):
            for j in range(window_size//2, stft_matrix.shape[1] - window_size//2):
                window = stft_matrix[i, j-window_size//2:j+window_size//2+1]
                smoothed[i, j] = np.median(window)
        return smoothed

    def voice_activity_detection(self, audio, sr, frame_length=2048, hop_length=512):
        """语音活动检测，识别人声片段"""
        # 计算短时能量
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        magnitude = np.abs(stft)
        energy = np.sum(magnitude**2, axis=0)
        
        # 计算谱质心（人声特征）
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        
        # 计算零交叉率
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 人声活动判断
        energy_threshold = np.percentile(energy, 30)
        centroid_min, centroid_max = 500, 3000  # 人声谱质心范围
        zcr_threshold = 0.3
        
        vad = (energy > energy_threshold) & \
              (spectral_centroids > centroid_min) & \
              (spectral_centroids < centroid_max) & \
              (zcr < zcr_threshold)
        
        return vad

    def run(self):
        try:
            self.status_updated.emit("正在加载音频文件...")
            song, sr = librosa.load(self.song_path, sr=None, mono=False)
            accompaniment, sr_acc = librosa.load(self.accompaniment_path, sr=None, mono=False)
            self.progress_updated.emit(10)

            # 音频预处理
            if song.ndim == 1: song = np.vstack([song, song])
            if accompaniment.ndim == 1: accompaniment = np.vstack([accompaniment, accompaniment])
            
            if sr != sr_acc:
                self.status_updated.emit("重采样伴奏以匹配歌曲采样率...")
                accompaniment = librosa.resample(accompaniment, orig_sr=sr_acc, target_sr=sr)
            
            min_len = min(song.shape[1], accompaniment.shape[1])
            song = song[:, :min_len]
            accompaniment = accompaniment[:, :min_len]
            
            self.progress_updated.emit(20)

            # 获取参数
            algorithm = self.params.get('algorithm', 'enhanced_spectral')
            strength = self.params.get('strength', 100) / 100.0
            n_fft = self.params.get('n_fft', 2048)
            hop_length = self.params.get('hop_length', 512)
            alpha = self.params.get('alpha', 200) / 100.0
            beta = self.params.get('beta', 5) / 100.0
            enable_vad = self.params.get('enable_vad', True)
            enable_harmonic = self.params.get('enable_harmonic', True)
            enable_temporal = self.params.get('enable_temporal', True)
            enable_denoising = self.params.get('enable_denoising', True)

            vocal_channels = []
            num_channels = song.shape[0]

            for i in range(num_channels):
                self.status_updated.emit(f"正在处理声道 {i+1}/{num_channels}...")
                
                song_ch = song[i]
                acc_ch = accompaniment[i]
                
                # 可选：谐波分离预处理
                if enable_harmonic:
                    self.status_updated.emit(f"声道 {i+1}: 进行谐波分离...")
                    song_ch = self.harmonic_percussive_separation(song_ch, sr)
                    acc_ch = self.harmonic_percussive_separation(acc_ch, sr)
                
                # STFT变换
                song_stft = librosa.stft(song_ch, n_fft=n_fft, hop_length=hop_length)
                acc_stft = librosa.stft(acc_ch, n_fft=n_fft, hop_length=hop_length)
                
                # 可选：时域平滑
                if enable_temporal:
                    self.status_updated.emit(f"声道 {i+1}: 应用时域平滑...")
                    acc_stft = self.temporal_smoothing(acc_stft)
                
                # 根据算法选择处理方式
                if algorithm == 'enhanced_spectral':
                    self.status_updated.emit(f"声道 {i+1}: 增强频谱减法...")
                    vocal_stft = self.spectral_subtraction_enhanced(
                        song_stft, acc_stft, 
                        alpha=alpha * strength, 
                        beta=beta
                    )
                    
                elif algorithm == 'adaptive_wiener':
                    self.status_updated.emit(f"声道 {i+1}: 自适应维纳滤波...")
                    vocal_stft = self.adaptive_wiener_filter(song_stft, acc_stft, strength=strength)
                    
                elif algorithm == 'multi_stage':
                    self.status_updated.emit(f"声道 {i+1}: 多级处理...")
                    # 第一级：频谱减法
                    stage1 = self.spectral_subtraction_enhanced(
                        song_stft, acc_stft, alpha=alpha * strength, beta=beta
                    )
                    # 第二级：维纳滤波优化
                    vocal_stft = self.adaptive_wiener_filter(song_stft, acc_stft, vocal_estimate=stage1, strength=strength)
                    
                else:  # 默认：混合方法
                    vocal_estimate = song_stft - (acc_stft * strength)
                    vocal_mag_estimate = np.abs(vocal_estimate)
                    acc_mag = np.abs(acc_stft)
                    mask = vocal_mag_estimate**2 / (vocal_mag_estimate**2 + acc_mag**2 + 1e-10)
                    vocal_stft = song_stft * mask
                
                # 转换回时域
                vocal_ch = librosa.istft(vocal_stft, hop_length=hop_length)
                
                # 可选：语音活动检测后处理
                if enable_vad:
                    self.status_updated.emit(f"声道 {i+1}: 应用语音活动检测...")
                    vad = self.voice_activity_detection(vocal_ch, sr, n_fft, hop_length)
                    
                    # 根据VAD结果调整音频
                    vad_expanded = np.repeat(vad, hop_length)[:len(vocal_ch)]
                    vocal_ch = vocal_ch * vad_expanded
                
                vocal_channels.append(vocal_ch)
                self.progress_updated.emit(30 + int(60 * (i + 1) / num_channels))

            self.status_updated.emit("合并声道并后处理...")
            vocal_stereo = np.vstack(vocal_channels)
            
            # 全局后处理
            if enable_denoising:
                self.status_updated.emit("应用降噪处理...")
                # 简单的低通滤波去除高频噪声
                for i in range(vocal_stereo.shape[0]):
                    vocal_stereo[i] = signal.sosfilt(
                        signal.butter(6, 8000, btype='low', fs=sr, output='sos'),
                        vocal_stereo[i]
                    )
            
            self.progress_updated.emit(95)
            self.status_updated.emit("保存文件...")
            sf.write(self.output_path, vocal_stereo.T, sr)
            
            self.progress_updated.emit(100)
            self.processing_finished.emit(self.output_path)
            
        except Exception as e:
            self.error_occurred.emit(f"处理过程中出错: {str(e)}")

    def stop(self):
        self.is_running = False


class EnhancedVocalExtractorApp(QMainWindow):
    """增强版人声提取应用程序"""

    def __init__(self):
        super().__init__()
        self.song_path = ""
        self.accompaniment_path = ""
        self.output_path = ""
        self.processing_thread = None
        self.settings = QSettings('VocalExtractor', 'EnhancedVersion')
        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle("增强版人声提取器 - 舞台录音专用")
        self.setGeometry(100, 100, 600, 500)

        main_layout = QVBoxLayout()
        
        # 文件选择区域
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()

        # 歌曲文件
        song_layout = QHBoxLayout()
        self.song_label = QLabel("舞台录音:")
        self.song_path_edit = QLineEdit()
        self.song_path_edit.setReadOnly(True)
        self.song_btn = QPushButton("浏览...")
        self.song_btn.clicked.connect(self.select_song)
        song_layout.addWidget(self.song_label)
        song_layout.addWidget(self.song_path_edit)
        song_layout.addWidget(self.song_btn)
        file_layout.addLayout(song_layout)

        # 伴奏文件
        acc_layout = QHBoxLayout()
        self.acc_label = QLabel("伴奏文件:")
        self.acc_path_edit = QLineEdit()
        self.acc_path_edit.setReadOnly(True)
        self.acc_btn = QPushButton("浏览...")
        self.acc_btn.clicked.connect(self.select_accompaniment)
        self.auto_find_btn = QPushButton("自动查找")
        self.auto_find_btn.clicked.connect(self.auto_find_accompaniment)
        acc_layout.addWidget(self.acc_label)
        acc_layout.addWidget(self.acc_path_edit)
        acc_layout.addWidget(self.acc_btn)
        acc_layout.addWidget(self.auto_find_btn)
        file_layout.addLayout(acc_layout)

        # 输出文件
        output_layout = QHBoxLayout()
        self.output_label = QLabel("输出文件:")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_btn = QPushButton("浏览...")
        self.output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_btn)
        file_layout.addLayout(output_layout)

        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # 参数设置区域
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout()

        # 算法选择
        algo_layout = QHBoxLayout()
        self.algo_label = QLabel("提取算法:")
        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "增强频谱减法 (推荐)",
            "自适应维纳滤波",
            "多级处理",
            "传统混合算法"
        ])
        self.algo_combo.setToolTip(
            "增强频谱减法: 专为舞台录音优化，最大程度去除垫音。\n"
            "自适应维纳滤波: 智能频率响应，适合复杂混音。\n"
            "多级处理: 结合多种算法，效果最佳但耗时较长。\n"
            "传统混合算法: 原有算法，作为对比。"
        )
        algo_layout.addWidget(self.algo_label)
        algo_layout.addWidget(self.algo_combo)
        params_layout.addLayout(algo_layout)

        # 强度控制
        strength_layout = QHBoxLayout()
        self.strength_label = QLabel("消除强度:")
        self.strength_value_label = QLabel("100%")
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(150)
        self.strength_slider.setValue(100)
        self.strength_slider.setTickPosition(QSlider.TicksBelow)
        self.strength_slider.setTickInterval(25)
        self.strength_slider.valueChanged.connect(self.update_strength_label)
        strength_layout.addWidget(self.strength_label)
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_value_label)
        params_layout.addLayout(strength_layout)

        # 设置按钮
        settings_layout = QHBoxLayout()
        self.settings_btn = QPushButton("高级设置...")
        self.settings_btn.clicked.connect(self.open_settings)
        settings_layout.addWidget(self.settings_btn)
        settings_layout.addStretch()
        params_layout.addLayout(settings_layout)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # 状态区域
        status_group = QGroupBox("处理状态")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("就绪 - 请选择舞台录音和对应伴奏")
        status_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始提取人声")
        self.start_btn.clicked.connect(self.start_processing)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_strength_label(self):
        value = self.strength_slider.value()
        self.strength_value_label.setText(f"{value}%")

    def select_song(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择舞台录音文件", "", 
            "音频文件 (*.mp3 *.wav *.flac *.m4a)"
        )
        if file_path:
            self.song_path = file_path
            self.song_path_edit.setText(file_path)
            # 自动设置输出路径
            dir_name = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            self.output_path = os.path.join(dir_name, f"{name}_vocals_enhanced.wav")
            self.output_path_edit.setText(self.output_path)
            # 自动查找伴奏
            self.auto_find_accompaniment()

    def select_accompaniment(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择伴奏文件", "", 
            "音频文件 (*.mp3 *.wav *.flac *.m4a)"
        )
        if file_path:
            self.accompaniment_path = file_path
            self.acc_path_edit.setText(file_path)

    def select_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存人声文件", self.output_path, 
            "音频文件 (*.wav)"
        )
        if file_path:
            self.output_path = file_path
            self.output_path_edit.setText(file_path)

    def auto_find_accompaniment(self):
        """自动查找匹配的伴奏文件"""
        if not self.song_path:
            QMessageBox.warning(self, "警告", "请先选择歌曲文件！")
            return
            
        dir_name = os.path.dirname(self.song_path)
        base_name = os.path.basename(self.song_path)
        name, ext = os.path.splitext(base_name)
        
        # 伴奏文件可能的关键词
        keywords = ["伴奏", "accompaniment", "instrumental", "inst", "karaoke", "off vocal", "minus one", "backing"]
        
        best_match = None
        highest_ratio = 0
        
        for file in os.listdir(dir_name):
            file_lower = file.lower()
            if file_lower.endswith((".mp3", ".wav", ".flac", ".m4a")) and file != base_name:
                file_name, file_ext = os.path.splitext(file_lower)
                
                # 计算文件名相似度
                ratio = SequenceMatcher(None, name.lower(), file_name).ratio()
                
                # 如果包含伴奏关键词，提高匹配度
                for keyword in keywords:
                    if keyword in file_name:
                        ratio += 0.3
                        break
                
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = os.path.join(dir_name, file)
        
        if best_match and highest_ratio > 0.4:
            self.accompaniment_path = best_match
            self.acc_path_edit.setText(best_match)
            self.status_label.setText(f"已自动找到可能的伴奏文件 (匹配度: {highest_ratio:.2f})")
        else:
            self.status_label.setText("未找到匹配的伴奏文件，请手动选择")

    def open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.status_label.setText("设置已更新")

    def get_current_settings(self):
        """获取当前设置"""
        settings = QSettings('VocalExtractor', 'EnhancedVersion')
        return {
            'n_fft': settings.value('fft_size', 2048, type=int),
            'hop_length': settings.value('hop_length', 512, type=int),
            'alpha': settings.value('alpha', 200, type=int),
            'beta': settings.value('beta', 5, type=int),
            'enable_harmonic': settings.value('enable_harmonic', True, type=bool),
            'enable_temporal': settings.value('enable_temporal', True, type=bool),
            'enable_vad': settings.value('enable_vad', True, type=bool),
            'enable_denoising': settings.value('enable_denoising', True, type=bool)
        }

    def start_processing(self):
        if not self.song_path or not self.accompaniment_path or not self.output_path:
            QMessageBox.warning(self, "警告", "请确保已选择舞台录音、伴奏和输出文件！")
            return

        # 算法映射
        algo_map = {
            "增强频谱减法 (推荐)": "enhanced_spectral",
            "自适应维纳滤波": "adaptive_wiener", 
            "多级处理": "multi_stage",
            "传统混合算法": "hybrid"
        }
        
        # 获取当前设置
        current_settings = self.get_current_settings()
        
        params = {
            'algorithm': algo_map[self.algo_combo.currentText()],
            'strength': self.strength_slider.value(),
            **current_settings
        }

        self.processing_thread = EnhancedProcessingThread(
            self.song_path, self.accompaniment_path, self.output_path, params
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.processing_finished.connect(self.processing_done)
        self.processing_thread.error_occurred.connect(self.processing_error)
        self.processing_thread.start()
        self.set_ui_enabled(False)

    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
            self.update_status("处理已取消")
            self.set_ui_enabled(True)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def processing_done(self, output_path):
        self.update_status(f"处理完成！增强人声已保存到: {output_path}")
        QMessageBox.information(
            self, "完成", 
            f"舞台人声提取完成！\n文件已保存到: {output_path}\n\n"
            "建议试听效果，如仍有残留可调整参数重新处理。"
        )
        self.set_ui_enabled(True)

    def processing_error(self, error_message):
        self.update_status(f"错误: {error_message}")
        QMessageBox.critical(self, "错误", error_message)
        self.set_ui_enabled(True)

    def set_ui_enabled(self, enabled):
        self.start_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)
        self.song_btn.setEnabled(enabled)
        self.acc_btn.setEnabled(enabled)
        self.output_btn.setEnabled(enabled)
        self.auto_find_btn.setEnabled(enabled)
        self.algo_combo.setEnabled(enabled)
        self.strength_slider.setEnabled(enabled)
        self.settings_btn.setEnabled(enabled)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("人声提取器2.0")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("VocalExtractor")
    
    window = EnhancedVocalExtractorApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
