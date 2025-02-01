import pygame
import random
import time
import math
import numpy as np
import cv2
from queue import Queue
import threading

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
ROAD_WIDTH = 100
VEHICLE_TYPES = {
    'car': {'length': 40, 'width': 20, 'color': (255, 255, 0), 'speed': 3},
    'bus': {'length': 60, 'width': 25, 'color': (0, 191, 255), 'speed': 2},  # Light blue
    'truck': {'length': 55, 'width': 25, 'color': (255, 165, 0), 'speed': 2}  # Orange
}

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
YELLOW = (255, 255, 0)

# Add to constants
MIN_VEHICLE_SPACING = 50  # Minimum space between vehicles
SIGNAL_MIN_TIME = 10  # Minimum time for a signal phase (seconds)
SIGNAL_MAX_TIME = 45  # Maximum time for a signal phase (seconds)
YELLOW_TIME = 3      # Yellow light duration (seconds)
DECORATION_TREES = [(100, 100), (700, 100), (100, 700), (700, 700)]  # Tree positions

# Add new colors
TREE_GREEN = (34, 139, 34)  # Dark green for trees
BUILDING_COLOR = (169, 169, 169)  # Light gray for buildings
YELLOW_LIGHT = (255, 191, 0)  # Yellow for traffic signal

# Add to constants
EMERGENCY_VEHICLE_TYPES = {
    'ambulance': {'length': 50, 'width': 25, 'color': (255, 0, 0), 'speed': 5},  # Red
    'police': {'length': 45, 'width': 22, 'color': (0, 0, 255), 'speed': 5},     # Blue
    'fire': {'length': 55, 'width': 25, 'color': (255, 69, 0), 'speed': 5}       # Red-Orange
}

class TrafficLight:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction
        self.state = 'red'
        self.timer = 0
        self.changing = False
        self.time_to_change = 0  # Add countdown timer
        
    def draw(self, screen):
        x, y = self.position
        # Draw traffic light pole and box
        pygame.draw.rect(screen, BLACK, (x-5, y-30, 10, 60))
        pygame.draw.rect(screen, BLACK, (x-15, y-30, 30, 90))
        
        # Draw the lights
        if self.state == 'red':
            pygame.draw.circle(screen, RED, (x, y-20), 10)
            pygame.draw.circle(screen, BLACK, (x, y), 10)
            pygame.draw.circle(screen, BLACK, (x, y+20), 10)
        elif self.state == 'yellow':
            pygame.draw.circle(screen, BLACK, (x, y-20), 10)
            pygame.draw.circle(screen, YELLOW_LIGHT, (x, y), 10)
            pygame.draw.circle(screen, BLACK, (x, y+20), 10)
        else:  # green
            pygame.draw.circle(screen, BLACK, (x, y-20), 10)
            pygame.draw.circle(screen, BLACK, (x, y), 10)
            pygame.draw.circle(screen, GREEN, (x, y+20), 10)
            
        # Draw countdown timer
        font = pygame.font.Font(None, 24)
        timer_text = f"{int(self.time_to_change)}s"
        timer_surface = font.render(timer_text, True, BLACK)
        screen.blit(timer_surface, (x-15, y+40))

def draw_tree(screen, position, size=40):
    x, y = position
    # Draw trunk
    pygame.draw.rect(screen, (139, 69, 19), (x-5, y-5, 10, 20))
    # Draw leaves
    pygame.draw.circle(screen, TREE_GREEN, (x, y-15), size//2)

def draw_building(screen, position, size):
    x, y = position
    pygame.draw.rect(screen, BUILDING_COLOR, (x, y, size, size))
    # Draw windows
    window_size = size // 4
    for i in range(2):
        for j in range(2):
            pygame.draw.rect(screen, (255, 255, 224),  # Light yellow windows
                           (x + i*window_size*2 + 5, 
                            y + j*window_size*2 + 5, 
                            window_size, window_size))

class Vehicle:
    def __init__(self, x, y, direction, vehicle_type='car'):
        self.x = x
        self.y = y
        self.direction = direction
        self.vehicle_type = vehicle_type
        self.stopped = False
        # Get vehicle properties from VEHICLE_TYPES
        props = VEHICLE_TYPES[vehicle_type]
        self.length = props['length']
        self.width = props['width']
        self.color = props['color']
        self.speed = props['speed']
        self.waiting_time = 0  # Track how long vehicle has been waiting
    
    def should_stop(self, traffic_lights):
        # Check if vehicle should stop at red light
        for light in traffic_lights:
            if light.state == 'red':
                if self.direction in ['left', 'right'] and light.direction == 'horizontal':
                    # Check if vehicle is near the light
                    light_x = light.position[0]
                    if self.direction == 'right' and self.x < light_x and self.x > light_x - 100:
                        return True
                    if self.direction == 'left' and self.x > light_x and self.x < light_x + 100:
                        return True
                elif self.direction in ['up', 'down'] and light.direction == 'vertical':
                    light_y = light.position[1]
                    if self.direction == 'down' and self.y < light_y and self.y > light_y - 100:
                        return True
                    if self.direction == 'up' and self.y > light_y and self.y < light_y + 100:
                        return True
        return False
    
    def check_collision(self, other_vehicles):
        # Check for potential collisions with other vehicles
        for other in other_vehicles:
            if other == self:
                continue
                
            # Calculate distance between vehicles
            dx = self.x - other.x
            dy = self.y - other.y
            distance = (dx * dx + dy * dy) ** 0.5
            
            # Check if vehicles are too close
            min_distance = MIN_VEHICLE_SPACING + (self.length + other.length) / 2
            
            if distance < min_distance:
                # If moving in same direction, adjust speed
                if self.direction == other.direction:
                    if ((self.direction in ['right', 'down'] and distance < min_distance) or
                        (self.direction in ['left', 'up'] and distance > -min_distance)):
                        return True
                # If at intersection, check cross traffic
                elif self.is_at_intersection() and other.is_at_intersection():
                    return True
            
        return False
    
    def is_at_intersection(self):
        # Check if vehicle is in the intersection area
        intersection_x = WINDOW_WIDTH // 2
        intersection_y = WINDOW_HEIGHT // 2
        intersection_size = ROAD_WIDTH * 2
        
        return (abs(self.x - intersection_x) < intersection_size and 
                abs(self.y - intersection_y) < intersection_size)
    
    def move(self, traffic_lights, other_vehicles):
        if self.should_stop(traffic_lights) or self.check_collision(other_vehicles):
            self.stopped = True
            self.waiting_time += 1
            return
        
        self.stopped = False
        self.waiting_time = 0
        
        # Move based on direction
        if self.direction == 'right':
            self.x += self.speed
        elif self.direction == 'left':
            self.x -= self.speed
        elif self.direction == 'up':
            self.y -= self.speed
        elif self.direction == 'down':
            self.y += self.speed
            
    def draw(self, screen):
        if self.direction in ['left', 'right']:
            rect = pygame.Rect(self.x, self.y, self.length, self.width)
        else:
            rect = pygame.Rect(self.x, self.y, self.width, self.length)
        pygame.draw.rect(screen, self.color, rect)
        # Draw vehicle type label
        font = pygame.font.Font(None, 20)
        label = font.render(self.vehicle_type, True, BLACK)
        screen.blit(label, (self.x, self.y - 15))

class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Traffic Simulation")
        
        # Create traffic lights
        self.traffic_lights = [
            TrafficLight((WINDOW_WIDTH//2 - ROAD_WIDTH, WINDOW_HEIGHT//2 - ROAD_WIDTH), 'horizontal'),
            TrafficLight((WINDOW_WIDTH//2 + ROAD_WIDTH, WINDOW_HEIGHT//2 + ROAD_WIDTH), 'horizontal'),
            TrafficLight((WINDOW_WIDTH//2 - ROAD_WIDTH, WINDOW_HEIGHT//2 + ROAD_WIDTH), 'vertical'),
            TrafficLight((WINDOW_WIDTH//2 + ROAD_WIDTH, WINDOW_HEIGHT//2 - ROAD_WIDTH), 'vertical')
        ]
        
        self.vehicles = []
        self.clock = pygame.time.Clock()
        self.light_timer = 0
        self.current_phase = 'horizontal'  # or 'vertical'
        
        # Add frame queue for sharing frames with the AI system
        self.frame_queue = Queue(maxsize=1)
        self.traffic_decision_queue = Queue(maxsize=1)
        
        # Add vehicle counts
        self.vehicle_counts = {
            'car': {'left': 0, 'right': 0},
            'bus': {'left': 0, 'right': 0},
            'truck': {'left': 0, 'right': 0}
        }
        
        self.phase_start_time = time.time()
        self.total_waiting_time = {'horizontal': 0, 'vertical': 0}
        self.yellow_start_time = 0
        self.switching_phase = False
        
        # Add analytics data
        self.analytics = {
            'total_vehicles_processed': 0,
            'average_wait_time': 0,
            'emergency_vehicles_count': 0,
            'peak_hour_data': {},
            'traffic_density_history': [],
            'signal_efficiency': 0,
            'congestion_points': [],
            'incidents': []
        }
        
        # Add emergency vehicle tracking
        self.emergency_vehicles = []
        self.emergency_mode = False
        
    def calculate_signal_phase(self):
        """Calculate which phase should get green signal based on traffic density and waiting time"""
        # Handle yellow light transition
        if self.switching_phase:
            current_time = time.time()
            if current_time - self.yellow_start_time >= YELLOW_TIME:
                self.switching_phase = False
                self.phase_start_time = current_time
                # Switch to the opposite phase
                return 'vertical' if self.current_phase == 'horizontal' else 'horizontal'
            return self.current_phase

        # Check for emergency vehicles first
        if self.emergency_vehicles:
            emergency_phase = self.handle_emergency_vehicle()
            if emergency_phase != self.current_phase:
                self.switching_phase = True
                self.yellow_start_time = time.time()
            return self.current_phase

        # Count vehicles in each direction
        horizontal_vehicles = sum(1 for v in self.vehicles if v.direction in ['left', 'right'])
        vertical_vehicles = sum(1 for v in self.vehicles if v.direction in ['up', 'down'])
        
        # Count waiting vehicles
        horizontal_waiting = sum(1 for v in self.vehicles 
                               if v.direction in ['left', 'right'] and v.waiting_time > 0)
        vertical_waiting = sum(1 for v in self.vehicles 
                                 if v.direction in ['up', 'down'] and v.waiting_time > 0)

        # Calculate priority scores
        horizontal_score = horizontal_vehicles + (horizontal_waiting * 2)  # Waiting vehicles count double
        vertical_score = vertical_vehicles + (vertical_waiting * 2)

        current_time = time.time()
        phase_duration = current_time - self.phase_start_time

        # Determine if we should switch signals
        should_switch = False

        # Force switch if maximum time elapsed
        if phase_duration > SIGNAL_MAX_TIME:
            should_switch = True
            print("Forcing signal switch due to max time")
        # Switch based on traffic conditions if minimum time elapsed
        elif phase_duration > SIGNAL_MIN_TIME:
            if self.current_phase == 'horizontal' and vertical_score > horizontal_score * 1.2:
                should_switch = True
                print(f"Switching to vertical: {vertical_score} > {horizontal_score}")
            elif self.current_phase == 'vertical' and horizontal_score > vertical_score * 1.2:
                should_switch = True
                print(f"Switching to horizontal: {horizontal_score} > {vertical_score}")

        # If should switch, start yellow light phase
        if should_switch and not self.switching_phase:
            self.switching_phase = True
            self.yellow_start_time = current_time
            print(f"Starting yellow phase at {current_time}")

        return self.current_phase

    def update_traffic_lights(self):
        try:
            current_time = time.time()
            phase_duration = current_time - self.phase_start_time
            
            # Calculate time to next change
            if self.switching_phase:
                time_to_change = YELLOW_TIME - (current_time - self.yellow_start_time)
            else:
                if phase_duration > SIGNAL_MIN_TIME:
                    time_to_change = SIGNAL_MAX_TIME - phase_duration
                else:
                    time_to_change = SIGNAL_MIN_TIME - phase_duration
            
            # Update light states and timers
            for light in self.traffic_lights:
                if self.switching_phase:
                    light.state = 'yellow'
                    light.time_to_change = time_to_change
                elif light.direction == self.current_phase:
                    light.state = 'green'
                    light.time_to_change = time_to_change
                else:
                    light.state = 'red'
                    light.time_to_change = time_to_change
                    
        except Exception as e:
            print(f"Error updating traffic lights: {str(e)}")
    
    def spawn_vehicle(self):
        if random.random() < 0.05:
            direction = random.choice(['right', 'left', 'up', 'down'])
            
            # Check if spawn point is clear
            spawn_x, spawn_y = 0, 0
            if direction == 'right':
                spawn_x, spawn_y = -50, WINDOW_HEIGHT//2 - ROAD_WIDTH//2
            elif direction == 'left':
                spawn_x, spawn_y = WINDOW_WIDTH + 50, WINDOW_HEIGHT//2 + ROAD_WIDTH//2
            elif direction == 'up':
                spawn_x, spawn_y = WINDOW_WIDTH//2 + ROAD_WIDTH//2, WINDOW_HEIGHT + 50
            else:  # down
                spawn_x, spawn_y = WINDOW_WIDTH//2 - ROAD_WIDTH//2, -50
            
            # Check if spawn area is clear
            for vehicle in self.vehicles:
                dx = vehicle.x - spawn_x
                dy = vehicle.y - spawn_y
                if (dx * dx + dy * dy) ** 0.5 < MIN_VEHICLE_SPACING:
                    return
            
            vehicle_type = random.choices(
                ['car', 'bus', 'truck'],
                weights=[0.7, 0.15, 0.15]
            )[0]
            
            self.vehicles.append(Vehicle(spawn_x, spawn_y, direction, vehicle_type))
    
    def get_frame(self):
        # Convert pygame surface to numpy array for YOLO
        view = pygame.surfarray.array3d(self.screen)
        # Convert from (width, height, 3) to (height, width, 3) and RGB to BGR
        view = view.transpose([1, 0, 2])
        view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        return view.copy()  # Return a copy to avoid memory issues
    
    def draw_environment(self):
        # Fill background
        self.screen.fill((144, 238, 144))  # Light green for grass
        
        # Draw buildings in corners
        draw_building(self.screen, (50, 50), 150)
        draw_building(self.screen, (WINDOW_WIDTH-200, 50), 150)
        draw_building(self.screen, (50, WINDOW_HEIGHT-200), 150)
        draw_building(self.screen, (WINDOW_WIDTH-200, WINDOW_HEIGHT-200), 150)
        
        # Draw roads
        pygame.draw.rect(self.screen, GRAY, (0, WINDOW_HEIGHT//2 - ROAD_WIDTH, WINDOW_WIDTH, ROAD_WIDTH*2))
        pygame.draw.rect(self.screen, GRAY, (WINDOW_WIDTH//2 - ROAD_WIDTH, 0, ROAD_WIDTH*2, WINDOW_HEIGHT))
        
        # Draw road markings
        for i in range(0, WINDOW_WIDTH, 50):
            pygame.draw.rect(self.screen, WHITE, (i, WINDOW_HEIGHT//2 - 2, 30, 4))
            pygame.draw.rect(self.screen, WHITE, (i, WINDOW_HEIGHT//2 + ROAD_WIDTH - 2, 30, 4))
        
        for i in range(0, WINDOW_HEIGHT, 50):
            pygame.draw.rect(self.screen, WHITE, (WINDOW_WIDTH//2 - 2, i, 4, 30))
            pygame.draw.rect(self.screen, WHITE, (WINDOW_WIDTH//2 + ROAD_WIDTH - 2, i, 4, 30))
        
        # Draw trees
        for pos in DECORATION_TREES:
            draw_tree(self.screen, pos)
    
    def count_vehicles(self):
        # Reset counts
        for v_type in self.vehicle_counts:
            self.vehicle_counts[v_type] = {'left': 0, 'right': 0}
        
        # Count vehicles on each side
        for vehicle in self.vehicles:
            side = 'left' if vehicle.x < WINDOW_WIDTH//2 else 'right'
            self.vehicle_counts[vehicle.vehicle_type][side] += 1
    
    def draw_stats(self):
        font = pygame.font.Font(None, 24)
        y_pos = 10
        for v_type in self.vehicle_counts:
            text = f"{v_type.title()}: Left={self.vehicle_counts[v_type]['left']} Right={self.vehicle_counts[v_type]['right']}"
            label = font.render(text, True, BLACK)
            self.screen.blit(label, (10, y_pos))
            y_pos += 25
    
    def handle_emergency_vehicle(self):
        """Give priority to emergency vehicles"""
        if not self.emergency_vehicles:
            return False
            
        # Find closest emergency vehicle to intersection
        closest = min(self.emergency_vehicles, 
                     key=lambda v: abs(v.x - WINDOW_WIDTH//2) + abs(v.y - WINDOW_HEIGHT//2))
        
        # Determine which phase should be green
        if closest.direction in ['left', 'right']:
            return 'horizontal'
        return 'vertical'
    
    def update_analytics(self):
        """Update real-time analytics"""
        current_hour = time.localtime().tm_hour
        
        # Update vehicle counts
        total_vehicles = len(self.vehicles)
        if total_vehicles > 0:
            avg_wait = sum(v.waiting_time for v in self.vehicles) / total_vehicles
            self.analytics['average_wait_time'] = avg_wait
        
        # Update peak hour data
        if current_hour not in self.analytics['peak_hour_data']:
            self.analytics['peak_hour_data'][current_hour] = 0
        self.analytics['peak_hour_data'][current_hour] += total_vehicles
        
        # Calculate traffic density
        density = total_vehicles / (WINDOW_WIDTH * WINDOW_HEIGHT)
        self.analytics['traffic_density_history'].append(density)
        
        # Calculate signal efficiency
        green_time = time.time() - self.phase_start_time
        vehicles_processed = self.analytics['total_vehicles_processed']
        if green_time > 0:
            self.analytics['signal_efficiency'] = vehicles_processed / green_time
        
        # Identify congestion points
        self.analytics['congestion_points'] = [
            (v.x, v.y) for v in self.vehicles if v.waiting_time > 30
        ]
    
    def get_analytics_data(self):
        """Return real-time analytics data"""
        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'traffic_stats': self.vehicle_counts,
            'current_signal_phase': self.current_phase,
            'signal_duration': time.time() - self.phase_start_time,
            'analytics': self.analytics,
            'emergency_mode': self.emergency_mode,
            'vehicle_counts': {
                'total': len(self.vehicles),
                'emergency': len(self.emergency_vehicles),
                'waiting': sum(1 for v in self.vehicles if v.waiting_time > 0)
            },
            'congestion_level': len(self.analytics['congestion_points']),
            'average_speed': sum(v.speed for v in self.vehicles) / len(self.vehicles) if self.vehicles else 0,
            'intersection_status': {
                'north': sum(1 for v in self.vehicles if v.direction == 'up'),
                'south': sum(1 for v in self.vehicles if v.direction == 'down'),
                'east': sum(1 for v in self.vehicles if v.direction == 'right'),
                'west': sum(1 for v in self.vehicles if v.direction == 'left')
            },
            'traffic_lights': [
                {
                    'direction': light.direction,
                    'state': light.state,
                    'time_to_change': int(light.time_to_change),
                    'position': 'N/S' if light.direction == 'vertical' else 'E/W'
                } for light in self.traffic_lights
            ]
        }
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Update
            self.spawn_vehicle()
            self.update_traffic_lights()
            
            # Move vehicles with collision checking
            for vehicle in self.vehicles[:]:
                vehicle.move(self.traffic_lights, [v for v in self.vehicles if v != vehicle])
                if (vehicle.x < -100 or vehicle.x > WINDOW_WIDTH + 100 or 
                    vehicle.y < -100 or vehicle.y > WINDOW_HEIGHT + 100):
                    self.vehicles.remove(vehicle)
            
            # Count vehicles
            self.count_vehicles()
            
            # Draw
            self.draw_environment()
            self.draw_stats()
            
            # Draw traffic lights
            for light in self.traffic_lights:
                light.draw(self.screen)
            
            # Draw vehicles
            for vehicle in self.vehicles:
                vehicle.draw(self.screen)
            
            # Put the current frame in the queue for AI processing
            current_frame = self.get_frame()
            if self.frame_queue.empty():
                try:
                    self.frame_queue.put_nowait(current_frame)
                except:
                    pass
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    sim = Simulation()
    sim.run() 