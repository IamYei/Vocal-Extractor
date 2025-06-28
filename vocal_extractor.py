#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人声提取应用程序
使用频率分析方法从歌曲中提取人声，并提供高质量音频处理
"""

import os
import sys
import time
import threading
import glob
import io
from difflib import SequenceMatcher
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QFileDialog, QProgressBar, QSlider, QVBoxLayout, 
                             QHBoxLayout, QWidget, QMessageBox, QGroupBox, QLineEdit,
                             QCheckBox, QComboBox, QTabWidget, QListWidget, QListWidgetItem,
                             QSplitter, QFrame, QSpinBox, QDoubleSpinBox, QRadioButton,
                             QButtonGroup, QToolButton, QMenu, QAction, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl, QBuffer, QIODevice
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# 音频可视化类
class AudioVisualizer(FigureCanvas):
    """音频波形和频谱可视化"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(AudioVisualizer, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        
    def plot_waveform(self, audio, sr, title="波形图"):
        """绘制音频波形"""
        self.axes.clear()
        time = np.arange(0, len(audio)) / sr
        self.axes.plot(time, audio)
        self.axes.set_title(title)
        self.axes.set_xlabel("时间 (秒)")
        self.axes.set_ylabel("振幅")
        self.fig.tight_layout()
        self.draw()
        
    def plot_spectrogram(self, audio, sr, title="频谱图"):
        """绘制音频频谱图"""
        self.axes.clear()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=self.axes)
        self.axes.set_title(title)
        self.fig.tight_layout()
        self.draw()


# 音频处理工具类
class AudioProcessor:
    """音频处理工具集"""
    
    @staticmethod
    def apply_highpass_filter(audio, sr, cutoff=100):
        """应用高通滤波器"""
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(5, normal_cutoff, btype='highpass')
        return signal.filtfilt(b, a, audio)
    
    @staticmethod
    def apply_lowpass_filter(audio, sr, cutoff=8000):
        """应用低通滤波器"""
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(5, normal_cutoff, btype='lowpass')
        return signal.filtfilt(b, a, audio)
    
    @staticmethod
    def apply_bandpass_filter(audio, sr, low_cutoff=300, high_cutoff=3400):
        """应用带通滤波器（人声范围）"""
        nyquist = 0.5 * sr
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = signal.butter(5, [low, high], btype='band')
        return signal.filtfilt(b, a, audio)
    
    @staticmethod
    def apply_noise_reduction(audio, sr, noise_reduce_strength=0.5):
        """应用噪声降低"""
        # 计算短时傅里叶变换
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        
        # 计算幅度谱
        magnitude = np.abs(stft)
        
        # 估计噪声谱
        noise_magnitude = np.mean(magnitude[:, :10], axis=1, keepdims=True)  # 使用前几帧估计噪声
        
        # 计算信噪比
        snr = magnitude / (noise_magnitude + 1e-10)
        
        # 创建维纳滤波器掩码
        mask = (snr**2 - noise_reduce_strength) / (snr**2 + 1 - noise_reduce_strength)
        mask = np.maximum(0, mask)  # 确保掩码值非负
        
        # 应用掩码
        stft_denoised = stft * mask
        
        # 转换回时域
        return librosa.istft(stft_denoised, hop_length=hop_length)
    
    @staticmethod
    def apply_vocal_enhancement(audio, sr):
        """增强人声（增强中频）"""
        # 使用均衡器增强人声频率范围
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        
        # 创建均衡器增益曲线（增强1kHz-4kHz范围）
        freq_bins = np.fft.rfftfreq(n_fft, 1/sr)
        gains = np.ones(len(freq_bins))
        
        # 增强人声频率范围
        vocal_mask = np.logical_and(freq_bins >= 1000, freq_bins <= 4000)
        gains[vocal_mask] = 1.3  # 增强30%
        
        # 应用增益到频谱上
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        magnitude_enhanced = magnitude * gains[:magnitude.shape[0], np.newaxis]
        stft_enhanced = magnitude_enhanced * np.exp(1j * phase)
        
        # 转换回时域
        return librosa.istft(stft_enhanced, hop_length=hop_length)
    
    @staticmethod
    def apply_compression(audio, threshold=-20, ratio=4.0, attack=0.01, release=0.1):
        """应用动态范围压缩"""
        # 将阈值从dB转换为线性
        threshold_linear = 10.0 ** (threshold / 20.0)
        
        # 计算音频的包络
        abs_audio = np.abs(audio)
        
        # 简化的压缩器实现
        compressed = np.zeros_like(audio)
        gain = 1.0
        for i in range(len(audio)):
            if abs_audio[i] > threshold_linear:
                # 压缩超过阈值的部分
                gain_target = (threshold_linear + (abs_audio[i] - threshold_linear) / ratio) / abs_audio[i]
            else:
                gain_target = 1.0
                
            # 应用攻击和释放时间
            if gain_target < gain:
                gain = gain_target + (gain - gain_target) * np.exp(-1.0 / (attack * sr))
            else:
                gain = gain_target + (gain - gain_target) * np.exp(-1.0 / (release * sr))
                
            compressed[i] = audio[i] * gain
            
        return compressed
    
    @staticmethod
    def normalize_audio(audio, target_dB=-3):
        """将音频归一化到目标dB"""
        # 计算当前RMS值
        rms = np.sqrt(np.mean(audio**2))
        
        # 计算目标RMS值（从dB转换）
        target_rms = 10 ** (target_dB / 20.0)
        
        # 计算增益并应用
        gain = target_rms / (rms + 1e-10)  # 避免除以零
        return audio * gain


# 音频预览对话框
class AudioPreviewDialog(QDialog):
    """音频预览对话框，用于在保存前预览和调整提取的人声"""
    
    def __init__(self, audio_data, sr, parent=None):
        super().__init__(parent)
        self.audio_data = audio_data
        self.sr = sr
        self.is_playing = False
        self.player = QMediaPlayer()
        self.player.stateChanged.connect(self.on_state_changed)
        self.initUI()
        
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle("音频预览")
        self.setGeometry(200, 200, 800, 500)
        
        layout = QVBoxLayout()
        
        # 音频可视化区域
        self.visualizer = AudioVisualizer(self, width=7, height=4)
        self.visualizer.plot_waveform(self.audio_data, self.sr, "人声波形图")
        layout.addWidget(self.visualizer)
        
        # 控制按钮区域
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_playback)
        controls_layout.addWidget(self.stop_btn)
        
        # 可视化切换按钮
        self.view_toggle_btn = QPushButton("切换到频谱图")
        self.view_toggle_btn.clicked.connect(self.toggle_visualization)
        controls_layout.addWidget(self.view_toggle_btn)
        
        # 后处理选项
        self.post_process_group = QGroupBox("后处理选项")
        post_process_layout = QVBoxLayout()
        
        # 降噪选项
        self.noise_reduction_cb = QCheckBox("降噪")
        self.noise_reduction_cb.stateChanged.connect(self.apply_post_processing)
        post_process_layout.addWidget(self.noise_reduction_cb)
        
        # 人声增强选项
        self.vocal_enhance_cb = QCheckBox("人声增强")
        self.vocal_enhance_cb.stateChanged.connect(self.apply_post_processing)
        post_process_layout.addWidget(self.vocal_enhance_cb)
        
        # 带通滤波器选项
        self.bandpass_cb = QCheckBox("带通滤波器 (300Hz-3400Hz)")
        self.bandpass_cb.stateChanged.connect(self.apply_post_processing)
        post_process_layout.addWidget(self.bandpass_cb)
        
        self.post_process_group.setLayout(post_process_layout)
        controls_layout.addWidget(self.post_process_group)
        
        layout.addLayout(controls_layout)
        
        # 按钮区域
        buttons_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        
        # 准备音频数据用于播放
        self.prepare_audio_for_playback()
        
    def prepare_audio_for_playback(self):
        """将NumPy数组转换为可播放的格式"""
        # 确保音频数据在-1到1之间
        audio_normalized = np.copy(self.audio_data)
        if np.max(np.abs(audio_normalized)) > 1.0:
            audio_normalized = audio_normalized / np.max(np.abs(audio_normalized))
        
        # 转换为16位整数
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        # 写入内存缓冲区
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, self.sr, format='WAV')
        buffer.seek(0)
        
        # 创建QBuffer
        self.qbuffer = QBuffer(self)
        self.qbuffer.open(QIODevice.WriteOnly)
        self.qbuffer.write(buffer.read())
        self.qbuffer.close()
        
        # 设置媒体内容
        self.qbuffer.open(QIODevice.ReadOnly)
        self.player.setMedia(QMediaContent(), self.qbuffer)
        
    def toggle_playback(self):
        """切换播放/暂停状态"""
        if self.is_playing:
            self.player.pause()
            self.play_btn.setText("播放")
        else:
            self.player.play()
            self.play_btn.setText("暂停")
        self.is_playing = not self.is_playing
        
    def stop_playback(self):
        """停止播放"""
        self.player.stop()
        self.play_btn.setText("播放")
        self.is_playing = False
        
    def on_state_changed(self, state):
        """播放器状态改变时的处理"""
        if state == QMediaPlayer.StoppedState:
            self.play_btn.setText("播放")
            self.is_playing = False
            
    def toggle_visualization(self):
        """切换波形图和频谱图"""
        if self.view_toggle_btn.text() == "切换到频谱图":
            self.visualizer.plot_spectrogram(self.audio_data, self.sr, "人声频谱图")
            self.view_toggle_btn.setText("切换到波形图")
        else:
            self.visualizer.plot_waveform(self.audio_data, self.sr, "人声波形图")
            self.view_toggle_btn.setText("切换到频谱图")
            
    def apply_post_processing(self):
        """应用后处理并更新预览"""
        # 停止当前播放
        self.stop_playback()
        
        # 从原始音频数据开始
        processed_audio = np.copy(self.audio_data)
        
        # 应用选中的后处理
        if self.noise_reduction_cb.isChecked():
            processed_audio = AudioProcessor.apply_noise_reduction(processed_audio, self.sr, 0.5)
            
        if self.vocal_enhance_cb.isChecked():
            processed_audio = AudioProcessor.apply_vocal_enhancement(processed_audio, self.sr)
            
        if self.bandpass_cb.isChecked():
            processed_audio = AudioProcessor.apply_bandpass_filter(processed_audio, self.sr, 300, 3400)
        
        # 更新音频数据和可视化
        self.audio_data = processed_audio
        if self.view_toggle_btn.text() == "切换到频谱图":
            self.visualizer.plot_waveform(self.audio_data, self.sr, "人声波形图")
        else:
            self.visualizer.plot_spectrogram(self.audio_data, self.sr, "人声频谱图")
            
        # 准备新的音频数据用于播放
        self.prepare_audio_for_playback()
        
    def get_processed_audio(self):
        """获取处理后的音频数据"""
        return self.audio_data, self.sr


class ProcessingThread(QThread):
    """处理音频的线程"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    processing_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    preview_ready = pyqtSignal(np.ndarray, int)  # 音频数据和采样率
    
    def __init__(self, song_path, accompaniment_path, output_path, params):
        super().__init__()
        self.song_path = song_path
        self.accompaniment_path = accompaniment_path
        self.output_path = output_path
        self.params = params
        self.is_running = True
        self.processor = AudioProcessor()
        
    def run(self):
        try:
            self.status_updated.emit("正在加载歌曲文件...")
            # 加载歌曲和伴奏
            song, sr = librosa.load(self.song_path, sr=None, mono=True)
            self.progress_updated.emit(10)
            
            # 预处理歌曲
            if self.params.get('apply_preprocessing', False):
                self.status_updated.emit("正在预处理歌曲...")
                if self.params.get('highpass_filter', False):
                    song = AudioProcessor.apply_highpass_filter(song, sr, self.params.get('highpass_cutoff', 80))
                if self.params.get('lowpass_filter', False):
                    song = AudioProcessor.apply_lowpass_filter(song, sr, self.params.get('lowpass_cutoff', 8000))
            
            self.progress_updated.emit(20)
            
            self.status_updated.emit("正在加载伴奏文件...")
            accompaniment, sr_acc = librosa.load(self.accompaniment_path, sr=None, mono=True)
            
            # 确保采样率一致
            if sr != sr_acc:
                self.status_updated.emit("重采样伴奏以匹配歌曲采样率...")
                accompaniment = librosa.resample(accompaniment, orig_sr=sr_acc, target_sr=sr)
            
            # 预处理伴奏
            if self.params.get('apply_preprocessing', False):
                self.status_updated.emit("正在预处理伴奏...")
                if self.params.get('highpass_filter', False):
                    accompaniment = AudioProcessor.apply_highpass_filter(accompaniment, sr, self.params.get('highpass_cutoff', 80))
                if self.params.get('lowpass_filter', False):
                    accompaniment = AudioProcessor.apply_lowpass_filter(accompaniment, sr, self.params.get('lowpass_cutoff', 8000))
            
            # 确保长度一致
            min_len = min(len(song), len(accompaniment))
            song = song[:min_len]
            accompaniment = accompaniment[:min_len]
            
            self.progress_updated.emit(30)
            self.status_updated.emit("正在进行频率分析...")
            
            # 转换到频域
            n_fft = self.params.get('n_fft', 2048)
            hop_length = self.params.get('hop_length', 512)
            
            song_stft = librosa.stft(song, n_fft=n_fft, hop_length=hop_length)
            accompaniment_stft = librosa.stft(accompaniment, n_fft=n_fft, hop_length=hop_length)
            
            # 计算幅度谱和相位谱
            song_mag = np.abs(song_stft)
            song_phase = np.angle(song_stft)
            accompaniment_mag = np.abs(accompaniment_stft)
            
            self.progress_updated.emit(50)
            self.status_updated.emit("正在提取人声...")
            
            # 使用频率分析提取人声
            # 根据强度参数调整伴奏的影响
            strength = self.params.get('strength', 50) / 100.0
            adjusted_accompaniment_mag = accompaniment_mag * strength
            
            # 计算掩码 - 改进的掩码计算方法
            if self.params.get('use_advanced_mask', True):
                # 使用软掩码（Soft Mask）
                mask = song_mag**2 / (song_mag**2 + adjusted_accompaniment_mag**2 + 1e-10)
                # 应用掩码平滑
                mask = np.maximum(0, mask)
                # 应用频率相关的掩码增强
                freq_bins = np.fft.rfftfreq(n_fft, 1/sr)[:mask.shape[0]]
                vocal_range_mask = np.logical_and(freq_bins >= 300, freq_bins <= 3400)
                mask[vocal_range_mask] = np.power(mask[vocal_range_mask], 0.8)  # 增强人声频率范围
            else:
                # 使用简单的减法掩码
                mask = np.maximum(0, song_mag - adjusted_accompaniment_mag) / (song_mag + 1e-10)
                mask = np.nan_to_num(mask)  # 处理除以零的情况
            
            # 应用掩码到原始歌曲的复数频谱上
            vocal_stft = song_stft * mask
            
            self.progress_updated.emit(70)
            self.status_updated.emit("正在转换回时域...")
            
            # 转换回时域
            vocal = librosa.istft(vocal_stft, hop_length=hop_length)
            
            # 后处理
            if self.params.get('apply_postprocessing', False):
                self.status_updated.emit("正在进行后处理...")
                
                # 降噪
                if self.params.get('noise_reduction', False):
                    self.status_updated.emit("正在应用降噪...")
                    vocal = AudioProcessor.apply_noise_reduction(vocal, sr, self.params.get('noise_reduce_strength', 0.5))
                
                # 人声增强
                if self.params.get('vocal_enhancement', False):
                    self.status_updated.emit("正在增强人声...")
                    vocal = AudioProcessor.apply_vocal_enhancement(vocal, sr)
                
                # 应用带通滤波器（人声范围）
                if self.params.get('bandpass_filter', False):
                    self.status_updated.emit("正在应用带通滤波器...")
                    vocal = AudioProcessor.apply_bandpass_filter(vocal, sr, 
                                                              self.params.get('bandpass_low', 300), 
                                                              self.params.get('bandpass_high', 3400))
                
                # 动态范围压缩
                if self.params.get('compression', False):
                    self.status_updated.emit("正在应用动态范围压缩...")
                    vocal = AudioProcessor.apply_compression(vocal, 
                                                          self.params.get('compression_threshold', -20),
                                                          self.params.get('compression_ratio', 4.0))
                
                # 归一化
                if self.params.get('normalization', False):
                    self.status_updated.emit("正在归一化音频...")
                    vocal = AudioProcessor.normalize_audio(vocal, self.params.get('normalization_target', -3))
            
            self.progress_updated.emit(90)
            
            # 发送预览信号
            if self.params.get('preview_before_save', False):
                self.preview_ready.emit(vocal, sr)
            
            # 保存结果
            self.status_updated.emit("正在保存人声文件...")
            sf.write(self.output_path, vocal, sr)
            
            self.progress_updated.emit(100)
            self.processing_finished.emit(self.output_path)
            
        except Exception as e:
            self.error_occurred.emit(f"处理过程中出错: {str(e)}")
            
    def stop(self):
        self.is_running = False


class VocalExtractorApp(QMainWindow):
    """人声提取应用程序的主窗口"""
    
    def __init__(self):
        super().__init__()
        self.song_path = ""
        self.accompaniment_path = ""
        self.output_path = ""
        self.processing_thread = None
        self.initUI()
        
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle("人声提取器")
        self.setGeometry(100, 100, 600, 400)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 文件选择区域
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        
        # 歌曲选择
        song_layout = QHBoxLayout()
        self.song_label = QLabel("歌曲文件:")
        self.song_path_edit = QLineEdit()
        self.song_path_edit.setReadOnly(True)
        self.song_btn = QPushButton("浏览...")
        self.song_btn.clicked.connect(self.select_song)
        song_layout.addWidget(self.song_label)
        song_layout.addWidget(self.song_path_edit)
        song_layout.addWidget(self.song_btn)
        file_layout.addLayout(song_layout)
        
        # 伴奏选择
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
        
        # 输出选择
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
        
        # 强度滑块
        strength_layout = QHBoxLayout()
        self.strength_label = QLabel("提取强度:")
        self.strength_value_label = QLabel("50%")
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(50)
        self.strength_slider.setTickPosition(QSlider.TicksBelow)
        self.strength_slider.setTickInterval(10)
        self.strength_slider.valueChanged.connect(self.update_strength_label)
        strength_layout.addWidget(self.strength_label)
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_value_label)
        params_layout.addLayout(strength_layout)
        
        # 预处理选项
        self.preprocessing_cb = QCheckBox("启用预处理")
        self.preprocessing_cb.setChecked(False)
        self.preprocessing_cb.stateChanged.connect(self.toggle_preprocessing_options)
        params_layout.addWidget(self.preprocessing_cb)
        
        # 预处理选项组
        self.preprocessing_group = QGroupBox("预处理选项")
        self.preprocessing_group.setEnabled(False)
        preprocessing_layout = QVBoxLayout()
        
        # 高通滤波器选项
        highpass_layout = QHBoxLayout()
        self.highpass_cb = QCheckBox("高通滤波器")
        self.highpass_cutoff_label = QLabel("截止频率:")
        self.highpass_cutoff_spin = QSpinBox()
        self.highpass_cutoff_spin.setRange(20, 500)
        self.highpass_cutoff_spin.setValue(80)
        self.highpass_cutoff_spin.setSuffix(" Hz")
        highpass_layout.addWidget(self.highpass_cb)
        highpass_layout.addWidget(self.highpass_cutoff_label)
        highpass_layout.addWidget(self.highpass_cutoff_spin)
        highpass_layout.addStretch(1)
        preprocessing_layout.addLayout(highpass_layout)
        
        # 低通滤波器选项
        lowpass_layout = QHBoxLayout()
        self.lowpass_cb = QCheckBox("低通滤波器")
        self.lowpass_cutoff_label = QLabel("截止频率:")
        self.lowpass_cutoff_spin = QSpinBox()
        self.lowpass_cutoff_spin.setRange(1000, 20000)
        self.lowpass_cutoff_spin.setValue(8000)
        self.lowpass_cutoff_spin.setSuffix(" Hz")
        lowpass_layout.addWidget(self.lowpass_cb)
        lowpass_layout.addWidget(self.lowpass_cutoff_label)
        lowpass_layout.addWidget(self.lowpass_cutoff_spin)
        lowpass_layout.addStretch(1)
        preprocessing_layout.addLayout(lowpass_layout)
        
        self.preprocessing_group.setLayout(preprocessing_layout)
        params_layout.addWidget(self.preprocessing_group)
        
        # 预览选项
        preview_layout = QHBoxLayout()
        self.preview_cb = QCheckBox("处理完成后预览")
        self.preview_cb.setChecked(True)
        preview_layout.addWidget(self.preview_cb)
        params_layout.addLayout(preview_layout)
        
        # 高级选项
        advanced_layout = QVBoxLayout()
        self.advanced_cb = QCheckBox("使用高级掩码算法")
        self.advanced_cb.setChecked(True)
        advanced_layout.addWidget(self.advanced_cb)
        
        # 后处理选项
        self.postprocessing_cb = QCheckBox("启用后处理")
        self.postprocessing_cb.setChecked(False)
        self.postprocessing_cb.stateChanged.connect(self.toggle_postprocessing_options)
        advanced_layout.addWidget(self.postprocessing_cb)
        
        # 后处理选项组
        self.postprocessing_group = QGroupBox("后处理选项")
        self.postprocessing_group.setEnabled(False)
        postprocessing_layout = QVBoxLayout()
        
        # 降噪选项
        noise_layout = QHBoxLayout()
        self.noise_reduction_cb = QCheckBox("降噪")
        self.noise_strength_label = QLabel("强度:")
        self.noise_strength_slider = QSlider(Qt.Horizontal)
        self.noise_strength_slider.setRange(1, 10)
        self.noise_strength_slider.setValue(5)
        self.noise_strength_value = QLabel("0.5")
        self.noise_strength_slider.valueChanged.connect(self.update_noise_strength_label)
        noise_layout.addWidget(self.noise_reduction_cb)
        noise_layout.addWidget(self.noise_strength_label)
        noise_layout.addWidget(self.noise_strength_slider)
        noise_layout.addWidget(self.noise_strength_value)
        postprocessing_layout.addLayout(noise_layout)
        
        # 人声增强选项
        self.vocal_enhancement_cb = QCheckBox("人声增强")
        postprocessing_layout.addWidget(self.vocal_enhancement_cb)
        
        # 带通滤波器选项
        bandpass_layout = QHBoxLayout()
        self.bandpass_filter_cb = QCheckBox("带通滤波器")
        self.bandpass_low_label = QLabel("低频:")
        self.bandpass_low_spin = QSpinBox()
        self.bandpass_low_spin.setRange(100, 1000)
        self.bandpass_low_spin.setValue(300)
        self.bandpass_low_spin.setSuffix(" Hz")
        self.bandpass_high_label = QLabel("高频:")
        self.bandpass_high_spin = QSpinBox()
        self.bandpass_high_spin.setRange(2000, 8000)
        self.bandpass_high_spin.setValue(3400)
        self.bandpass_high_spin.setSuffix(" Hz")
        bandpass_layout.addWidget(self.bandpass_filter_cb)
        bandpass_layout.addWidget(self.bandpass_low_label)
        bandpass_layout.addWidget(self.bandpass_low_spin)
        bandpass_layout.addWidget(self.bandpass_high_label)
        bandpass_layout.addWidget(self.bandpass_high_spin)
        postprocessing_layout.addLayout(bandpass_layout)
        
        # 动态范围压缩选项
        compression_layout = QHBoxLayout()
        self.compression_cb = QCheckBox("动态范围压缩")
        self.compression_threshold_label = QLabel("阈值:")
        self.compression_threshold_spin = QSpinBox()
        self.compression_threshold_spin.setRange(-60, 0)
        self.compression_threshold_spin.setValue(-20)
        self.compression_threshold_spin.setSuffix(" dB")
        self.compression_ratio_label = QLabel("比率:")
        self.compression_ratio_spin = QDoubleSpinBox()
        self.compression_ratio_spin.setRange(1.0, 20.0)
        self.compression_ratio_spin.setValue(4.0)
        self.compression_ratio_spin.setSingleStep(0.1)
        compression_layout.addWidget(self.compression_cb)
        compression_layout.addWidget(self.compression_threshold_label)
        compression_layout.addWidget(self.compression_threshold_spin)
        compression_layout.addWidget(self.compression_ratio_label)
        compression_layout.addWidget(self.compression_ratio_spin)
        postprocessing_layout.addLayout(compression_layout)
        
        # 归一化选项
        normalization_layout = QHBoxLayout()
        self.normalization_cb = QCheckBox("音频归一化")
        self.normalization_target_label = QLabel("目标电平:")
        self.normalization_target_spin = QSpinBox()
        self.normalization_target_spin.setRange(-20, 0)
        self.normalization_target_spin.setValue(-3)
        self.normalization_target_spin.setSuffix(" dB")
        normalization_layout.addWidget(self.normalization_cb)
        normalization_layout.addWidget(self.normalization_target_label)
        normalization_layout.addWidget(self.normalization_target_spin)
        normalization_layout.addStretch(1)
        postprocessing_layout.addLayout(normalization_layout)
        
        self.postprocessing_group.setLayout(postprocessing_layout)
        advanced_layout.addWidget(self.postprocessing_group)
        
        params_layout.addLayout(advanced_layout)
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # 状态和进度区域
        status_group = QGroupBox("处理状态")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("就绪")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self.start_processing)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)
        
        # 设置中心窗口部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def update_strength_label(self):
        """更新强度标签"""
        value = self.strength_slider.value()
        self.strength_value_label.setText(f"{value}%")
        
    def update_noise_strength_label(self):
        """更新降噪强度标签"""
        value = self.noise_strength_slider.value() / 10.0
        self.noise_strength_value.setText(f"{value:.1f}")
        
    def select_song(self):
        """选择歌曲文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择歌曲文件", "", "音频文件 (*.mp3 *.wav *.flac *.m4a)")
        if file_path:
            self.song_path = file_path
            self.song_path_edit.setText(file_path)
            
            # 自动设置输出路径
            dir_name = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            self.output_path = os.path.join(dir_name, f"{name}_vocals{ext}")
            self.output_path_edit.setText(self.output_path)
            
            # 尝试自动查找伴奏
            self.auto_find_accompaniment()
            
    def select_accompaniment(self):
        """选择伴奏文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择伴奏文件", "", "音频文件 (*.mp3 *.wav *.flac *.m4a)")
        if file_path:
            self.accompaniment_path = file_path
            self.acc_path_edit.setText(file_path)
            
    def select_output(self):
        """选择输出文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存人声文件", self.output_path, "音频文件 (*.wav)")
        if file_path:
            self.output_path = file_path
            self.output_path_edit.setText(file_path)
            
    def auto_find_accompaniment(self):
        """自动查找伴奏文件"""
        if not self.song_path:
            QMessageBox.warning(self, "警告", "请先选择歌曲文件！")
            return
            
        dir_name = os.path.dirname(self.song_path)
        base_name = os.path.basename(self.song_path)
        name, ext = os.path.splitext(base_name)
        
        # 可能的伴奏文件关键词
        keywords = ["伴奏", "accompaniment", "instrumental", "inst", "karaoke", "off vocal", "minus one"]
        
        best_match = None
        highest_ratio = 0
        
        # 搜索同一目录下的所有音频文件
        for file in os.listdir(dir_name):
            file_lower = file.lower()
            if file_lower.endswith((".mp3", ".wav", ".flac", ".m4a")) and file != base_name:
                # 检查文件名是否包含伴奏关键词
                file_name, file_ext = os.path.splitext(file_lower)
                
                # 计算文件名相似度
                ratio = SequenceMatcher(None, name.lower(), file_name).ratio()
                
                # 如果文件名包含关键词，增加相似度权重
                for keyword in keywords:
                    if keyword in file_name:
                        ratio += 0.2
                        break
                        
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = os.path.join(dir_name, file)
        
        if best_match and highest_ratio > 0.4:  # 设置一个阈值
            self.accompaniment_path = best_match
            self.acc_path_edit.setText(best_match)
            self.status_label.setText(f"已自动找到可能的伴奏文件")
        else:
            self.status_label.setText("未找到匹配的伴奏文件，请手动选择")
            
    def toggle_preprocessing_options(self, state):
        """启用或禁用预处理选项"""
        self.preprocessing_group.setEnabled(state == Qt.Checked)
        
    def toggle_postprocessing_options(self, state):
        """启用或禁用后处理选项"""
        self.postprocessing_group.setEnabled(state == Qt.Checked)
        
    def start_processing(self):
        """开始处理"""
        if not self.song_path:
            QMessageBox.warning(self, "警告", "请选择歌曲文件！")
            return
            
        if not self.accompaniment_path:
            QMessageBox.warning(self, "警告", "请选择伴奏文件！")
            return
            
        if not self.output_path:
            QMessageBox.warning(self, "警告", "请选择输出文件！")
            return
            
        # 收集处理参数
        params = {
            'strength': self.strength_slider.value(),
            'use_advanced_mask': self.advanced_cb.isChecked(),
            'preview_before_save': self.preview_cb.isChecked(),
            
            # 预处理参数
            'apply_preprocessing': self.preprocessing_cb.isChecked(),
            'highpass_filter': self.highpass_cb.isChecked(),
            'highpass_cutoff': self.highpass_cutoff_spin.value(),
            'lowpass_filter': self.lowpass_cb.isChecked(),
            'lowpass_cutoff': self.lowpass_cutoff_spin.value(),
            
            # 后处理参数
            'apply_postprocessing': self.postprocessing_cb.isChecked(),
            'noise_reduction': self.noise_reduction_cb.isChecked(),
            'vocal_enhancement': self.vocal_enhancement_cb.isChecked(),
            'bandpass_filter': self.bandpass_filter_cb.isChecked(),
            'compression': self.compression_cb.isChecked(),
            'normalization': self.normalization_cb.isChecked(),
            
            # 技术参数
            'n_fft': 2048,
            'hop_length': 512,
            'noise_reduce_strength': self.noise_strength_slider.value() / 10.0,
            'bandpass_low': self.bandpass_low_spin.value(),
            'bandpass_high': self.bandpass_high_spin.value(),
            'compression_threshold': self.compression_threshold_spin.value(),
            'compression_ratio': self.compression_ratio_spin.value(),
            'normalization_target': self.normalization_target_spin.value()
        }
        
        # 创建并启动处理线程
        self.processing_thread = ProcessingThread(
            self.song_path, self.accompaniment_path, self.output_path, params)
        
        # 连接信号
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.processing_finished.connect(self.processing_done)
        self.processing_thread.error_occurred.connect(self.processing_error)
        self.processing_thread.preview_ready.connect(self.show_preview)
        
        # 启动线程
        self.processing_thread.start()
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.song_btn.setEnabled(False)
        self.acc_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        self.auto_find_btn.setEnabled(False)
        
    def cancel_processing(self):
        """取消处理"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
            self.update_status("处理已取消")
            self.reset_ui()
            
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """更新状态标签"""
        self.status_label.setText(message)
        
    def show_preview(self, audio_data, sr):
        """显示音频预览对话框"""
        self.update_status("处理完成，正在显示预览...")
        
        # 创建预览对话框
        preview_dialog = AudioPreviewDialog(audio_data, sr, self)
        result = preview_dialog.exec_()
        
        if result == QDialog.Accepted:
            # 用户点击了保存按钮，获取处理后的音频
            processed_audio, sr = preview_dialog.get_processed_audio()
            
            # 保存处理后的音频
            self.update_status("正在保存处理后的音频...")
            try:
                sf.write(self.output_path, processed_audio, sr)
                self.processing_done(self.output_path, False)  # 已经预览过，不需要再显示消息框
            except Exception as e:
                self.processing_error(f"保存音频时出错: {str(e)}")
        else:
            # 用户取消了保存
            self.update_status("预览已取消，未保存音频")
            self.reset_ui()
    
    def processing_done(self, output_path, show_message=True):
        """处理完成"""
        self.update_status(f"处理完成！人声已保存到: {output_path}")
        if show_message:
            QMessageBox.information(self, "完成", f"人声提取完成！\n文件已保存到: {output_path}")
        self.reset_ui()
        
    def processing_error(self, error_message):
        """处理错误"""
        self.update_status(f"错误: {error_message}")
        QMessageBox.critical(self, "错误", error_message)
        self.reset_ui()
        
    def reset_ui(self):
        """重置UI状态"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.song_btn.setEnabled(True)
        self.acc_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.auto_find_btn.setEnabled(True)
        

def main():
    app = QApplication(sys.argv)
    window = VocalExtractorApp()
    window.show()
    sys.exit(app.exec_())
    

if __name__ == "__main__":
    main()