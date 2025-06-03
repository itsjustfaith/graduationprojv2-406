#!/usr/bin/env python3
import socket
import threading
import ipaddress
import requests
from concurrent.futures import ThreadPoolExecutor
import netifaces

# Common IP camera ports and endpoints
CAMERA_PORTS = [80, 554, 8080, 8081, 8000, 443, 8443]
CAMERA_PATHS = [
    '/onvif/device_service',
    '/cgi-bin/hi3510/param.cgi',
    '/axis-cgi/mjpg/video.cgi',
    '/videostream.cgi',
    '/video.cgi',
    '/mjpg/video.mjpg',
    '/snap.jpg',
    '/image.jpg'
]

def get_local_network():
    """Get the local network range"""
    try:
        # Get default gateway interface
        gateways = netifaces.gateways()
        default_interface = gateways['default'][netifaces.AF_INET][1]
        
        # Get network info for the interface
        addrs = netifaces.ifaddresses(default_interface)
        if netifaces.AF_INET in addrs:
            ip_info = addrs[netifaces.AF_INET][0]
            ip = ip_info['addr']
            netmask = ip_info['netmask']
            
            # Calculate network
            network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
            return network
    except:
        # Fallback to common private networks
        return ipaddress.IPv4Network('192.168.1.0/24')

def check_port(ip, port, timeout=2):
    """Check if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((str(ip), port))
        sock.close()
        return result == 0
    except:
        return False

def check_http_response(ip, port, path, timeout=3):
    """Check HTTP response for camera-specific content"""
    try:
        protocols = ['http', 'https'] if port in [443, 8443] else ['http']
        
        for protocol in protocols:
            url = f"{protocol}://{ip}:{port}{path}"
            response = requests.get(url, timeout=timeout, verify=False)
            
            # Look for camera-specific indicators
            content = response.text.lower()
            headers = str(response.headers).lower()
            
            camera_indicators = [
                'camera', 'webcam', 'ipcam', 'hikvision', 'dahua', 
                'axis', 'onvif', 'rtsp', 'mjpeg', 'video', 'surveillance'
            ]
            
            for indicator in camera_indicators:
                if indicator in content or indicator in headers:
                    return True
                    
        return False
    except:
        return False

def scan_ip(ip):
    """Scan a single IP for camera services"""
    open_ports = []
    camera_detected = False
    
    # Check common camera ports
    for port in CAMERA_PORTS:
        if check_port(ip, port):
            open_ports.append(port)
            
            # Check for camera-specific HTTP responses
            for path in CAMERA_PATHS:
                if check_http_response(ip, port, path):
                    camera_detected = True
                    break
    
    if camera_detected or (open_ports and any(port in [554, 8080, 8081] for port in open_ports)):
        return {
            'ip': str(ip),
            'ports': open_ports,
            'likely_camera': camera_detected
        }
    
    return None

def main():
    print("Scanning local network for IP cameras...")
    
    # Get local network range
    network = get_local_network()
    print(f"Scanning network: {network}")
    
    cameras_found = []
    
    # Use ThreadPoolExecutor for concurrent scanning
    with ThreadPoolExecutor(max_workers=50) as executor:
        # Submit all IP addresses for scanning
        futures = {executor.submit(scan_ip, ip): ip for ip in network.hosts()}
        
        # Collect results
        for future in futures:
            result = future.result()
            if result:
                cameras_found.append(result)
                print(f"Found potential camera at {result['ip']} - Ports: {result['ports']}")
    
    print(f"\nScan complete! Found {len(cameras_found)} potential IP cameras:")
    print("-" * 50)
    
    for camera in cameras_found:
        status = "Confirmed Camera" if camera['likely_camera'] else "Possible Camera"
        print(f"IP: {camera['ip']} - {status}")
        print(f"Open Ports: {', '.join(map(str, camera['ports']))}")
        print("-" * 30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScan interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install requests netifaces")