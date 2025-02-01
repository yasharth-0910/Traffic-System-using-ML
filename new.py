import cv2
import time
import threading
import numpy as np
from queue import Queue
from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO
from traffic_simulation import Simulation

app = Flask(__name__)

# Global variables for the latest annotated frame and a lock for thread safety
outputFrame = None
lock = threading.Lock()

# Configuration parameters
CONFIG = {
    'VEHICLE_CLASSES': [2, 3, 5, 7],  # COCO classes: car, motorbike, bus, truck
    'CONFIDENCE_THRESHOLD': 0.5,
    'MIN_SIGNAL_DURATION': 30,  # minimum seconds for a signal phase
    'PROCESSING_INTERVAL': 0.1,  # seconds between processing frames
}

# Enhanced traffic statistics
traffic_stats = {
    'left_count': 0,
    'right_count': 0,
    'signal': 'LEFT',
    'signal_duration': 0,
    'last_switch_time': time.time(),
    'vehicle_counts': {
        'car': {'left': 0, 'right': 0},
        'bus': {'left': 0, 'right': 0},
        'truck': {'left': 0, 'right': 0}
    }
}

# Load YOLO v8 model (make sure you have the ultralytics package and model weights)
# You might use a model such as 'yolov8n.pt' (nano version) for faster processing.
model = YOLO('yolov8n.pt')

def video_processing(simulation):
    """
    Process frames from the simulation instead of a video capture
    """
    global outputFrame, traffic_stats
    
    try:
        while True:
            try:
                # Get frame from simulation with timeout
                if simulation.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame = simulation.frame_queue.get()
                if frame is None:
                    continue
                
                # Ensure frame is in the correct format
                if not isinstance(frame, np.ndarray):
                    print("Invalid frame format")
                    continue
                
                # Run YOLO detection
                results = model(frame, verbose=False)
                detections = results[0].boxes if results else None

                left_count = 0
                right_count = 0
                (h, w) = frame.shape[:2]

                if detections is not None:
                    for box in detections:
                        coords = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = coords
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls in CONFIG['VEHICLE_CLASSES'] and conf > CONFIG['CONFIDENCE_THRESHOLD']:
                            cx = (x1 + x2) / 2
                            if cx < w / 2:
                                left_count += 1
                            else:
                                right_count += 1
                            
                            # Draw detection box
                            color = (0, 255, 0)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            label = f"{model.names[cls]}: {conf:.2f}"
                            cv2.putText(frame, label, (int(x1), int(y1)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Update traffic statistics
                current_time = time.time()
                elapsed_time = current_time - traffic_stats['last_switch_time']
                
                if elapsed_time >= CONFIG['MIN_SIGNAL_DURATION']:
                    left_ratio = left_count / (right_count + 1)
                    right_ratio = right_count / (left_count + 1)
                    
                    new_signal = traffic_stats['signal']
                    if left_ratio > 1.5 and traffic_stats['signal'] == 'RIGHT':
                        new_signal = 'LEFT'
                        traffic_stats['last_switch_time'] = current_time
                    elif right_ratio > 1.5 and traffic_stats['signal'] == 'LEFT':
                        new_signal = 'RIGHT'
                        traffic_stats['last_switch_time'] = current_time
                    
                    # Send decision to simulation
                    if new_signal != traffic_stats['signal']:
                        traffic_stats['signal'] = new_signal
                        if simulation.traffic_decision_queue.empty():
                            try:
                                simulation.traffic_decision_queue.put_nowait(new_signal)
                            except:
                                pass

                traffic_stats['left_count'] = left_count
                traffic_stats['right_count'] = right_count
                traffic_stats['signal_duration'] = int(elapsed_time)

                # Annotate frame
                cv2.putText(frame, f"Left Count: {left_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Right Count: {right_count}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Signal: {traffic_stats['signal']} ({traffic_stats['signal_duration']}s)", 
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Update traffic statistics
                traffic_stats['vehicle_counts'] = simulation.vehicle_counts

                with lock:
                    outputFrame = frame.copy()

                time.sleep(CONFIG['PROCESSING_INTERVAL'])
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                time.sleep(0.1)
                continue
            
    except Exception as e:
        print(f"Video processing thread error: {str(e)}")

def generate():
    global outputFrame
    while True:
        try:
            with lock:
                if outputFrame is None:
                    continue
                
                # Ensure frame is in the correct format
                if not isinstance(outputFrame, np.ndarray):
                    continue
                    
                # Convert frame to JPEG
                flag, encodedImage = cv2.imencode(".jpg", outputFrame)
                if not flag:
                    continue
                    
            # Yield the output frame in byte format
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            
        except Exception as e:
            print(f"Frame generation error: {str(e)}")
            time.sleep(0.1)
            continue
        
        time.sleep(0.1)

# Flask route: Admin GUI home page
@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
      <head>
        <title>National Traffic Management System</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          .stats { 
            background: #f0f0f0; 
            padding: 15px; 
            border-radius: 5px;
            margin-top: 20px;
          }
          .signal {
            font-weight: bold;
            font-size: 1.2em;
          }
          .vehicle-counts {
            margin-top: 15px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
          }
          .vehicle-type {
            background: white;
            padding: 10px;
            border-radius: 5px;
          }
          .analytics-panel {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
          }
          .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          }
          .emergency {
            background: #ffebee;
            color: #c62828;
            font-weight: bold;
          }
        </style>
      </head>
      <body>
        <h1>National Traffic Management System</h1>
        <div>
          <img src="{{ url_for('video_feed') }}" width="800" />
        </div>
        <div class="stats" id="stats"></div>
        <div class="analytics-panel" id="analytics"></div>
        
        <script>
          function updateAnalytics() {
            fetch('/api/v1/traffic-data')
              .then(response => response.json())
              .then(data => {
                document.getElementById('analytics').innerHTML = `
                  <div class="metric-card">
                    <h3>Traffic Density</h3>
                    <p>Total Vehicles: ${data.vehicle_counts.total}</p>
                    <p>Waiting Vehicles: ${data.vehicle_counts.waiting}</p>
                    <p>Average Speed: ${data.average_speed.toFixed(2)} units/s</p>
                  </div>
                  <div class="metric-card ${data.emergency_mode ? 'emergency' : ''}">
                    <h3>Emergency Status</h3>
                    <p>Emergency Mode: ${data.emergency_mode ? 'ACTIVE' : 'Inactive'}</p>
                    <p>Emergency Vehicles: ${data.vehicle_counts.emergency}</p>
                  </div>
                  <div class="metric-card">
                    <h3>Intersection Status</h3>
                    <p>North: ${data.intersection_status.north} vehicles</p>
                    <p>South: ${data.intersection_status.south} vehicles</p>
                    <p>East: ${data.intersection_status.east} vehicles</p>
                    <p>West: ${data.intersection_status.west} vehicles</p>
                  </div>
                  <div class="metric-card">
                    <h3>Performance Metrics</h3>
                    <p>Signal Efficiency: ${data.analytics.signal_efficiency.toFixed(2)}</p>
                    <p>Average Wait Time: ${data.analytics.average_wait_time.toFixed(2)}s</p>
                    <p>Congestion Level: ${data.congestion_level}</p>
                  </div>
                `;
              });
          }
          
          setInterval(updateAnalytics, 1000);
          updateAnalytics();
          
          function updateStats() {
            fetch('/stats')
              .then(response => response.json())
              .then(data => {
                let vehicleHtml = '<div class="vehicle-counts">';
                for (const [type, counts] of Object.entries(data.vehicle_counts)) {
                  vehicleHtml += `
                    <div class="vehicle-type">
                      <h3>${type.charAt(0).toUpperCase() + type.slice(1)}</h3>
                      <p>Left: ${counts.left}</p>
                      <p>Right: ${counts.right}</p>
                    </div>
                  `;
                }
                vehicleHtml += '</div>';
                
                document.getElementById('stats').innerHTML = `
                  <h2>Traffic Statistics</h2>
                  <p class="signal">Current Signal: ${data.signal} (${data.signal_duration}s)</p>
                  <h3>Vehicle Counts</h3>
                  ${vehicleHtml}
                `;
              })
              .catch(error => console.error('Error fetching stats:', error));
          }
          setInterval(updateStats, 1000);
          updateStats();
        </script>
      </body>
    </html>
    """)

# Flask route: stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route: return traffic statistics as JSON
@app.route('/stats')
def stats():
    return jsonify(traffic_stats)

# Add new routes
@app.route('/api/v1/traffic-data')
def traffic_data():
    """Real-time traffic data API endpoint"""
    return jsonify(sim.get_analytics_data())

@app.route('/api/v1/historical-data')
def historical_data():
    """Historical traffic data API endpoint"""
    return jsonify({
        'peak_hours': sim.analytics['peak_hour_data'],
        'density_history': sim.analytics['traffic_density_history'],
        'efficiency_metrics': {
            'signal_efficiency': sim.analytics['signal_efficiency'],
            'average_wait_time': sim.analytics['average_wait_time']
        }
    })

@app.route('/api/v1/emergency')
def emergency_status():
    """Emergency vehicle status API endpoint"""
    return jsonify({
        'emergency_mode': sim.emergency_mode,
        'emergency_vehicles': len(sim.emergency_vehicles),
        'priority_direction': sim.current_phase if sim.emergency_mode else None
    })

if __name__ == '__main__':
    try:
        print("Starting simulation...")
        sim = Simulation()
        sim_thread = threading.Thread(target=sim.run, daemon=True)
        sim_thread.start()
        print("Simulation thread started")
        
        print("Starting video processing...")
        t = threading.Thread(target=video_processing, args=(sim,), daemon=True)
        t.start()
        print("Video processing thread started")
        
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Startup error: {str(e)}")
