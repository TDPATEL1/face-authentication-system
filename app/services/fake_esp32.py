from fastapi import FastAPI

app = FastAPI(title="Fake ESP32 Door Controller")


door_unlocked = False


@app.post("/unlock")
def unlock():
    global door_unlocked

    door_unlocked = True

    print("FAKE ESP32: DOOR UNLOCKED")

    return {
        "success": True,
        "message": "Door unlocked",
        "unlocked": True,
    }


@app.post("/lock")
def lock():
    global door_unlocked

    door_unlocked = False

    print("FAKE ESP32: DOOR LOCKED")

    return {
        "success": True,
        "message": "Door locked",
        "unlocked": False,
    }


@app.get("/status")
def status():
    return {
        "unlocked": door_unlocked,
    }