import threading
import time
import csv
import os

try:
    from jtop import jtop
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False

class HardwareMonitor:
    """背景側錄硬體資源消耗 (基於 jtop，專為 Jetson 設計)"""
    def __init__(self, output_csv="hardware_stats.csv", interval=1.0):
        self.output_csv = output_csv
        self.interval = interval
        self.is_running = False
        self.thread = None
        self.jetson = None
        if HAS_JTOP:
            self.jetson = jtop()
        else:
            print("[Warning] jtop is not installed or accessible. Hardware Monitor will be disabled.")

    def start(self):
        if not HAS_JTOP or not self.jetson:
            return
            
        try:
            self.jetson.start()
            self.is_running = True
            self.thread = threading.Thread(target=self._monitor_loop)
            self.thread.daemon = True
            self.thread.start()
        except Exception as e:
            print(f"[Warning] Failed to start jtop. Ensure container has access to Jetson stats socket: {e}")
            self.is_running = False

    def _monitor_loop(self):
        try:
            with open(self.output_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                # csv 標頭
                writer.writerow(['Timestamp', 'CPU_Usage(%)', 'GPU_Usage(%)', 'RAM_Used(MB)', 'Power(mW)', 'GPU_Temp(C)'])
                
                while self.is_running and self.jetson.ok():
                    try:
                        writer.writerow([
                            time.time(),
                            self.jetson.cpu['total']['user'] if 'total' in self.jetson.cpu else 0, # CPU 整體使用率
                            self.jetson.stats['GPU'] if 'GPU' in self.jetson.stats else 0,           # GPU 使用率
                            round((self.jetson.memory['RAM']['used'] / 1024.0), 2) if 'RAM' in self.jetson.memory else 0,# 記憶體使用量 (KB 轉 MB)
                            self.jetson.power['tot']['power'] if 'tot' in self.jetson.power else 0,# 總功耗
                            self.jetson.temperature['gpu']['temp'] if 'gpu' in self.jetson.temperature else 0   # GPU 溫度
                        ])
                    except KeyError:
                        pass # Ignore minor stat sync issues
                    time.sleep(self.interval)
        except Exception as e:
            print(f"[Warning] Hardware monitor loop stopped unexpectedly: {e}")

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            
        if self.jetson and self.jetson.ok():
            self.jetson.close()
            
        if HAS_JTOP and os.path.exists(self.output_csv):
            print(f"硬體資源紀錄已匯出至 {self.output_csv}")
