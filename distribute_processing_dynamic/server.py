import socket
import pickle
import random
from threading import Thread
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Constants
CLASS_NAMES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle"}
COLOR_MAP = {
    0: (255, 0, 0),      # Red for Person
    1: (0, 255, 255),    # Cyan for Bicycle
    2: (0, 255, 0),      # Green for Car
    3: (0, 0, 255),      # Blue for Motorcycle
}

def broadcast_message_to_clients(message, clients):
    available_clients = []
    for client_socket, addr in clients:
        client_socket.sendall(pickle.dumps(message))
        response = client_socket.recv(1024).decode("utf-8")
        if response.lower() == "yes":
            available_clients.append((client_socket, addr))
    return available_clients

def assign_objects_to_clients(available_clients):
    object_ids = list(CLASS_NAMES.keys())
    random.shuffle(object_ids)
    object_assignment = {}

    for idx, (client_socket, addr) in enumerate(available_clients):
        assigned_objects = object_ids[idx::len(available_clients)]  # Spread objects across clients
        object_assignment[client_socket] = assigned_objects

        # Send the length of the pickled object assignment first
        assigned_objects_pickle = pickle.dumps(assigned_objects)
        data_length = len(assigned_objects_pickle)
        client_socket.sendall(data_length.to_bytes(8, byteorder="big"))  # Send data length
        client_socket.sendall(assigned_objects_pickle)  # Send actual data

        print(f"Assigned {assigned_objects} to client {addr}")

    return object_assignment

# Image Communication Functions
def send_image_to_client(client_socket, image_data):
    """Send image data to the client."""
    try:
        image_data_pickle = pickle.dumps(image_data)
        data_length = len(image_data_pickle)
        # Send the length of the image data first
        client_socket.sendall(data_length.to_bytes(8, byteorder="big"))
        # Send the actual image data
        client_socket.sendall(image_data_pickle)
        print(f"Sent image data of size {data_length} bytes to client.")
    except Exception as e:
        print(f"Failed to send image data: {e}")

def receive_detection_from_client(client_socket):
    """Receive object detection results from a client."""
    data = b""
    while True:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet
    return pickle.loads(data)

# Client Handling
def handle_client(client_socket, addr, image_data, results, object_assignment):
    """Handle a single client connection and process detection."""
    try:
        assigned_objects = object_assignment[client_socket]

        # Now send the image data after the objects have already been assigned
        send_image_to_client(client_socket, image_data)

        # Receive detection results from the client
        client_results = receive_detection_from_client(client_socket)
        for obj_type in assigned_objects:
            results[obj_type] = client_results.get(obj_type, [])
            print(f"Received {CLASS_NAMES[obj_type]} results from client {addr}")

    except ConnectionResetError:
        print(f"Client {addr} closed the connection unexpectedly.")
    except Exception as e:
        print(f"Error while handling client {addr}: {e}")
    finally:
        client_socket.close()

# Server Initialization
def start_server(image_path):
    """Initialize the server, manage client connections, and aggregate results."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("192.168.1.9", 8095))  
    server_socket.listen(4)  # Listen for up to 4 clients

    print("Server is waiting for clients to connect...")

    # Load the image data
    with open(image_path, "rb") as f:
        image_data = f.read()

    clients = []

    # Accept connections from clients
    while len(clients) < 4:
        client_socket, addr = server_socket.accept()
        clients.append((client_socket, addr))
        print(f"Client {addr} connected")

    while True:
        # Query the user
        user_input = input("Run detection? (yes/no): ").strip().lower()
        if user_input == "yes":
            # Broadcast message to all clients and check availability
            available_clients = broadcast_message_to_clients("Check availability", clients)
            if not available_clients:
                print("No clients are available.")
                break

            # Randomly assign objects to available clients
            object_assignment = assign_objects_to_clients(available_clients)

            # Gather results from all available clients
            results = {}
            for client_socket, addr in available_clients:
                client_thread = Thread(target=handle_client, args=(client_socket, addr, image_data, results, object_assignment))
                client_thread.start()

            for client_socket, addr in available_clients:
                client_thread.join()

            # Display the final results
            display_image_with_detections(image_path, results, "detected_objects_image.jpg")
            print("Final aggregated results:", results)
            break

    server_socket.close()

# Image Processing and Visualization
def display_image_with_detections(image_path, results, output_path):
    """Display the image with object detections and save it to a file."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Iterate through each object's detections and draw bounding boxes
    for object_type, detections in results.items():
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  
            confidence = detection.conf[0].item()
            color = COLOR_MAP.get(object_type)
            # Draw bounding box and label on the image
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1), f"{CLASS_NAMES[object_type]}: {confidence:.2f}", fill=color)

    image.save(output_path)
    print(f"Image saved to {output_path}")
    plt.imshow(image)
    plt.axis('off')
    plt.show()

#================== Example Usage
if __name__ == "__main__":
    image_path = "Fog_project/image.jpg"  
    start_server(image_path)
