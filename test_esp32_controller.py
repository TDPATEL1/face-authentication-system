from app.services.esp32_controller import ESP32DoorController


controller = ESP32DoorController(
    "http://127.0.0.1:9000"
)


print("Initial status:")
print(controller.is_unlocked())

print("\nUnlocking...")
print(controller.unlock())

print("\nStatus after unlock:")
print(controller.is_unlocked())

print("\nLocking...")
print(controller.lock())

print("\nStatus after lock:")
print(controller.is_unlocked())