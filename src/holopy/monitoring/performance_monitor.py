from typing import Dict, Optional, List
import numpy as np
import time
import psutil
import threading
from collections import deque
from dataclasses import dataclass
from ..config.constants import (
    MONITORING_INTERVAL,
    HISTORY_LENGTH,
    ALERT_THRESHOLDS
)
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Container for system performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    evolution_rate: float
    accuracy_metrics: Dict[str, float]
    alerts: List[str]

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(
        self,
        monitoring_interval: float = MONITORING_INTERVAL,
        history_length: int = HISTORY_LENGTH
    ):
        self.monitoring_interval = monitoring_interval
        self.history_length = history_length
        
        # Initialize metric storage
        self.metrics_history = deque(maxlen=history_length)
        self.alert_history = deque(maxlen=history_length)
        
        # Initialize monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        logger.info("Initialized PerformanceMonitor")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        try:
            if not self.is_monitoring:
                self.is_monitoring = True
                self.monitor_thread = threading.Thread(
                    target=self._monitoring_loop
                )
                self.monitor_thread.start()
                logger.info("Started performance monitoring")
            
        except Exception as e:
            logger.error(f"Monitor start failed: {str(e)}")
            raise
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        try:
            if self.is_monitoring:
                self.is_monitoring = False
                self.monitor_thread.join()
                logger.info("Stopped performance monitoring")
            
        except Exception as e:
            logger.error(f"Monitor stop failed: {str(e)}")
            raise
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.is_monitoring:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Check for alerts
                alerts = self._check_alerts(metrics)
                
                # Store results
                system_metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_usage=metrics['cpu_usage'],
                    memory_usage=metrics['memory_usage'],
                    evolution_rate=metrics['evolution_rate'],
                    accuracy_metrics=metrics['accuracy_metrics'],
                    alerts=alerts
                )
                
                self.metrics_history.append(system_metrics)
                
                # Log alerts
                if alerts:
                    logger.warning(
                        f"Performance alerts: {', '.join(alerts)}"
                    )
                
                # Wait for next interval
                time.sleep(self.monitoring_interval)
            
        except Exception as e:
            logger.error(f"Monitoring loop failed: {str(e)}")
            self.is_monitoring = False
            raise
    
    def _collect_metrics(self) -> Dict:
        """Collect current system metrics."""
        try:
            return {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.Process().memory_info().rss,
                'evolution_rate': self._calculate_evolution_rate(),
                'accuracy_metrics': self._collect_accuracy_metrics()
            }
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {str(e)}")
            raise
    
    def _check_alerts(
        self,
        metrics: Dict
    ) -> List[str]:
        """Check for performance alerts."""
        try:
            alerts = []
            
            # Check CPU usage
            if metrics['cpu_usage'] > ALERT_THRESHOLDS['cpu_usage']:
                alerts.append(f"High CPU usage: {metrics['cpu_usage']}%")
            
            # Check memory usage
            if metrics['memory_usage'] > ALERT_THRESHOLDS['memory_usage']:
                alerts.append(
                    f"High memory usage: {metrics['memory_usage']/1e6:.1f} MB"
                )
            
            # Check evolution rate
            if metrics['evolution_rate'] < ALERT_THRESHOLDS['min_evolution_rate']:
                alerts.append(
                    f"Low evolution rate: {metrics['evolution_rate']:.1f} steps/s"
                )
            
            # Check accuracy
            for metric, value in metrics['accuracy_metrics'].items():
                if value > ALERT_THRESHOLDS[f'max_{metric}']:
                    alerts.append(f"High {metric}: {value:.2e}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Alert check failed: {str(e)}")
            raise 