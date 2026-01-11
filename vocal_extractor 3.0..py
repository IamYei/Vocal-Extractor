#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#作者:不归bgi BiliBili:不归bgi
"""
人声提取&舞台消音应用程序
使用频率分析方法从歌曲中提取人声，支持立体声处理
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
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import tempfile

# 音频预览对话框（简化版，只有播放器）
class AudioPreviewDialog(QDialog):
    """音频预览对话框，用于在保存前预览提取的人声"""
    
    def __init__(self, audio_data=None, sr=None, file_path=None, parent=None):
        super().__init__(parent)
        self.audio_data = audio_data
        self.sr = sr
        self.file_path = file_path
        self.is_playing = False
        self.temp_file_path = None
        self.is_temp = False
        
        self.player = QMediaPlayer()
        self.player.setVolume(100)
        self.player.stateChanged.connect(self.on_state_changed)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        
        self.initUI()
        self.prepare_audio_for_playback()
        
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle("音频预览")
        self.resize(500, 150)
        self.setFixedSize(500, 180)
        
        layout = QVBoxLayout()
        
        # 顶部提示
        msg = "正在预览文件..." if self.file_path else "提取完成！您可以播放预览，确认满意后保存。"
        info_label = QLabel(msg)
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        # 进度条和时间
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("0:00")
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.sliderPressed.connect(self.player.pause)
        self.position_slider.sliderReleased.connect(self.check_play_status)
        self.total_time_label = QLabel("0:00")
        
        time_layout.addWidget(self.current_time_label)
        time_layout.addWidget(self.position_slider)
        time_layout.addWidget(self.total_time_label)
        layout.addLayout(time_layout)
        
        # 控制按钮
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_playback)
        controls_layout.addWidget(self.stop_btn)
        
        controls_layout.addStretch()
        controls_layout.addWidget(QLabel("音量:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.valueChanged.connect(self.player.setVolume)
        controls_layout.addWidget(self.volume_slider)
        
        layout.addLayout(controls_layout)
        
        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # 底部按钮
        buttons_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("关闭" if self.file_path else "放弃")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        if not self.file_path:
            self.save_btn = QPushButton("保存文件")
            self.save_btn.setDefault(True)
            self.save_btn.clicked.connect(self.accept)
            buttons_layout.addWidget(self.save_btn)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        
    def prepare_audio_for_playback(self):
        """准备音频播放"""
        try:
            if self.file_path:
                # 直接播放文件
                url = QUrl.fromLocalFile(self.file_path)
                content = QMediaContent(url)
                self.player.setMedia(content)
                self.is_temp = False
            else:
                # 创建临时文件
                fd, path = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                self.temp_file_path = path
                self.is_temp = True
                
                # 写入音频数据
                audio_clipped = np.clip(self.audio_data, -1.0, 1.0)
                sf.write(self.temp_file_path, audio_clipped.T, self.sr)
                
                # 加载到播放器
                url = QUrl.fromLocalFile(self.temp_file_path)
                content = QMediaContent(url)
                self.player.setMedia(content)
            
            # 自动播放
            self.player.play()
            self.is_playing = True
            self.play_btn.setText("暂停")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法准备预览音频: {str(e)}")

    def cleanup_temp_file(self):
        """清理临时文件"""
        if self.is_temp and self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.remove(self.temp_file_path)
            except:
                pass
            self.temp_file_path = None

    def check_play_status(self):
        if self.is_playing:
            self.player.play()

    def toggle_playback(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()
        self.is_playing = (self.player.state() == QMediaPlayer.PlayingState)
        
    def stop_playback(self):
        self.player.stop()
        self.is_playing = False
        
    def on_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setText("暂停")
            self.is_playing = True
        else:
            self.play_btn.setText("播放")
            if state == QMediaPlayer.StoppedState:
                self.is_playing = False
            
    def on_position_changed(self, position):
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position)
        self.current_time_label.setText(self.format_time(position))
        
    def on_duration_changed(self, duration):
        self.position_slider.setRange(0, duration)
        self.total_time_label.setText(self.format_time(duration))
        
    def set_position(self, position):
        self.player.setPosition(position)
        self.current_time_label.setText(self.format_time(position))
        
    def format_time(self, ms):
        s = ms // 1000
        m = s // 60
        s = s % 60
        return f"{m}:{s:02d}"
            
    def get_processed_audio(self):
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
        
    def process_channel(self, song_channel, accompaniment_channel, sr, n_fft, hop_length, strength, algorithm):
        """处理单个声道"""
        # 转换到频域
        song_stft = librosa.stft(song_channel, n_fft=n_fft, hop_length=hop_length)
        accompaniment_stft = librosa.stft(accompaniment_channel, n_fft=n_fft, hop_length=hop_length)
        
        # 计算幅度谱
        song_mag = np.abs(song_stft)
        accompaniment_mag = np.abs(accompaniment_stft)
        
        # 根据强度参数调整伴奏的影响
        adjusted_accompaniment_mag = accompaniment_mag * strength
        
        if algorithm == 'soft_mask':
            # 原算法：软掩码
            mask = song_mag**2 / (song_mag**2 + adjusted_accompaniment_mag**2 + 1e-10)
            mask = np.maximum(0, mask)
            
            # 应用掩码到原始歌曲的复数频谱上
            vocal_stft = song_stft * mask
            
        elif algorithm == 'spectral_subtraction':
            # 新算法：谱减法（使用频率计算，减少残留）
            vocal_mag = np.maximum(0, song_mag - adjusted_accompaniment_mag)
            # 保留原始相位
            phase = np.angle(song_stft)
            vocal_stft = vocal_mag * np.exp(1j * phase)
        
        # 转换回时域
        vocal = librosa.istft(vocal_stft, hop_length=hop_length)
        
        return vocal
        
    def run(self):
        try:
            self.status_updated.emit("正在加载歌曲文件...")
            # 加载歌曲和伴奏（保持立体声）
            song, sr = librosa.load(self.song_path, sr=None, mono=False)
            self.progress_updated.emit(10)
            
            self.status_updated.emit("正在加载伴奏文件...")
            accompaniment, sr_acc = librosa.load(self.accompaniment_path, sr=None, mono=False)
            
            # 确保采样率一致
            if sr != sr_acc:
                self.status_updated.emit("重采样伴奏以匹配歌曲采样率...")
                if len(accompaniment.shape) == 1:
                    accompaniment = librosa.resample(accompaniment, orig_sr=sr_acc, target_sr=sr)
                else:
                    accompaniment = np.array([
                        librosa.resample(accompaniment[0], orig_sr=sr_acc, target_sr=sr),
                        librosa.resample(accompaniment[1], orig_sr=sr_acc, target_sr=sr)
                    ])
            
            # 处理单声道和立体声的情况
            if len(song.shape) == 1:
                song = np.array([song, song])  # 转换为立体声
            if len(accompaniment.shape) == 1:
                accompaniment = np.array([accompaniment, accompaniment])  # 转换为立体声
                
            # 确保长度一致
            min_len = min(song.shape[1], accompaniment.shape[1])
            song = song[:, :min_len]
            accompaniment = accompaniment[:, :min_len]
            
            self.progress_updated.emit(30)
            self.status_updated.emit("正在进行频率分析...")
            
            # 处理参数
            n_fft = self.params.get('n_fft', 2048)
            hop_length = self.params.get('hop_length', 512)
            strength = self.params.get('strength', 50) / 100.0
            algorithm = self.params.get('algorithm', 'soft_mask')
            
            # 分别处理左右声道
            self.status_updated.emit("正在提取人声（左声道）...")
            vocal_left = self.process_channel(song[0], accompaniment[0], sr, n_fft, hop_length, strength, algorithm)
            self.progress_updated.emit(60)
            
            self.status_updated.emit("正在提取人声（右声道）...")
            vocal_right = self.process_channel(song[1], accompaniment[1], sr, n_fft, hop_length, strength, algorithm)
            self.progress_updated.emit(80)
            
            # 合并立体声
            vocal = np.array([vocal_left, vocal_right])
            
            # 裁剪任何NaN或inf值
            vocal = np.nan_to_num(vocal, nan=0.0, posinf=0.0, neginf=0.0)
            vocal = np.clip(vocal, -1.0, 1.0)
            
            self.progress_updated.emit(90)
            
            # 发送预览信号
            if self.params.get('preview_before_save', False):
                self.preview_ready.emit(vocal, sr)
            else:
                # 直接保存
                self.status_updated.emit("正在保存人声文件...")
                sf.write(self.output_path, vocal.T, sr)
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
        self.setWindowTitle("人声提取器 v3.0")
        self.setGeometry(100, 100, 600, 400)  # 稍稍增加高度以容纳新控件
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 文件选择区域
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        
        # 歌曲选择
        song_layout = QHBoxLayout()
        self.song_label = QLabel("舞台音频:")
        self.song_path_edit = QLineEdit()
        self.song_path_edit.setReadOnly(True)
        self.song_btn = QPushButton("浏览...")
        self.song_btn.clicked.connect(self.select_song)
        self.preview_song_btn = QPushButton("预览")
        self.preview_song_btn.clicked.connect(lambda: self.preview_input_file(self.song_path))
        song_layout.addWidget(self.song_label)
        song_layout.addWidget(self.song_path_edit)
        song_layout.addWidget(self.song_btn)
        song_layout.addWidget(self.preview_song_btn)
        file_layout.addLayout(song_layout)
        
        # 伴奏选择
        acc_layout = QHBoxLayout()
        self.acc_label = QLabel("音源音频:")
        self.acc_path_edit = QLineEdit()
        self.acc_path_edit.setReadOnly(True)
        self.acc_btn = QPushButton("浏览...")
        self.acc_btn.clicked.connect(self.select_accompaniment)
        self.preview_acc_btn = QPushButton("预览")
        self.preview_acc_btn.clicked.connect(lambda: self.preview_input_file(self.accompaniment_path))
        self.auto_find_btn = QPushButton("自动查找: 开")
        self.auto_find_btn.setCheckable(True)
        self.auto_find_btn.setChecked(True)
        self.auto_find_btn.clicked.connect(self.toggle_auto_find)
        acc_layout.addWidget(self.acc_label)
        acc_layout.addWidget(self.acc_path_edit)
        acc_layout.addWidget(self.acc_btn)
        acc_layout.addWidget(self.preview_acc_btn)
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
        
        # 算法选择
        algorithm_layout = QHBoxLayout()
        self.algorithm_label = QLabel("提取算法:")
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("软掩码 (默认，平衡)", "soft_mask")
        self.algorithm_combo.addItem("谱减法 (减少残留，使用频率计算)", "spectral_subtraction")
        algorithm_layout.addWidget(self.algorithm_label)
        algorithm_layout.addWidget(self.algorithm_combo)
        params_layout.addLayout(algorithm_layout)
        
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
        
        # 预览选项
        preview_layout = QHBoxLayout()
        self.preview_cb = QCheckBox("处理完成后预览")
        self.preview_cb.setChecked(True)
        preview_layout.addWidget(self.preview_cb)
        preview_layout.addStretch()
        params_layout.addLayout(preview_layout)
        
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
        
    def toggle_auto_find(self):
        """切换自动查找状态"""
        if self.auto_find_btn.isChecked():
            self.auto_find_btn.setText("自动查找: 开")
            # 如果刚刚开启，尝试查找一次
            if self.song_path:
                self.auto_find_accompaniment()
        else:
            self.auto_find_btn.setText("自动查找: 关")

    def select_song(self):
        """选择舞台音频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择舞台音频", "", "音频文件 (*.mp3 *.wav *.flac *.m4a)")
        if file_path:
            self.song_path = file_path
            self.song_path_edit.setText(file_path)
            
            # 自动设置输出路径
            dir_name = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            self.output_path = os.path.join(dir_name, f"{name}_vocals{ext}")
            self.output_path_edit.setText(self.output_path)
            
            # 尝试自动查找伴奏 (如果开启)
            if self.auto_find_btn.isChecked():
                self.auto_find_accompaniment()
            
    def select_accompaniment(self):
        """选择音源音频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音源音频", "", "音频文件 (*.mp3 *.wav *.flac *.m4a)")
        if file_path:
            self.accompaniment_path = file_path
            self.acc_path_edit.setText(file_path)
            
    def preview_input_file(self, path):
        """预览输入文件"""
        if not path:
            QMessageBox.warning(self, "提示", "请先选择文件！")
            return
            
        if not os.path.exists(path):
            QMessageBox.warning(self, "错误", "文件不存在！")
            return
            
        preview_dialog = AudioPreviewDialog(file_path=path, parent=self)
        preview_dialog.exec_()
            
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
            'preview_before_save': self.preview_cb.isChecked(),
            'algorithm': self.algorithm_combo.currentData(),
            'n_fft': 2048,
            'hop_length': 512
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
                sf.write(self.output_path, processed_audio.T, sr)
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