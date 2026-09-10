#include <WiFi.h>
#include <WebServer.h>

// ==================================================
// Wi-Fi Configuration
// ==================================================

const char* WIFI_SSID = "Prizor_AITECH";
const char* WIFI_PASSWORD = "qwerty@54321";

// ==================================================
// Hardware
// ==================================================

const int DOOR_LED_PIN = 2;

// ==================================================
// Web Server
// ==================================================

WebServer server(80);

// Door state
bool doorUnlocked = false;


// ==================================================
// Helper: JSON Response
// ==================================================

void sendJsonResponse(
    int statusCode,
    const String& message
) {
    String json = "{";
    json += "\"success\":true,";
    json += "\"message\":\"";
    json += message;
    json += "\",";
    json += "\"unlocked\":";
    json += doorUnlocked ? "true" : "false";
    json += "}";

    server.send(
        statusCode,
        "application/json",
        json
    );
}


// ==================================================
// Unlock Door
// ==================================================

void handleUnlock() {

    doorUnlocked = true;

    digitalWrite(
        DOOR_LED_PIN,
        HIGH
    );

    Serial.println("DOOR UNLOCKED");

    sendJsonResponse(
        200,
        "Door unlocked"
    );
}


// ==================================================
// Lock Door
// ==================================================

void handleLock() {

    doorUnlocked = false;

    digitalWrite(
        DOOR_LED_PIN,
        LOW
    );

    Serial.println("DOOR LOCKED");

    sendJsonResponse(
        200,
        "Door locked"
    );
}


// ==================================================
// Door Status
// ==================================================

void handleStatus() {

    String json = "{";
    json += "\"unlocked\":";
    json += doorUnlocked ? "true" : "false";
    json += "}";

    server.send(
        200,
        "application/json",
        json
    );
}


// ==================================================
// Not Found
// ==================================================

void handleNotFound() {

    server.send(
        404,
        "application/json",
        "{\"success\":false,\"message\":\"Endpoint not found\"}"
    );
}


// ==================================================
// Setup
// ==================================================

void setup() {

    Serial.begin(115200);

    delay(1000);

    // Configure LED
    pinMode(
        DOOR_LED_PIN,
        OUTPUT
    );

    digitalWrite(
        DOOR_LED_PIN,
        LOW
    );

    Serial.println();
    Serial.println("==============================");
    Serial.println("ESP32 Door Controller");
    Serial.println("==============================");

    // ------------------------------------------------
    // Connect Wi-Fi
    // ------------------------------------------------

    WiFi.begin(
        WIFI_SSID,
        WIFI_PASSWORD
    );

    Serial.print("Connecting to Wi-Fi");

    while (WiFi.status() != WL_CONNECTED) {

        delay(500);

        Serial.print(".");
    }

    Serial.println();
    Serial.println("Wi-Fi connected!");

    Serial.print("ESP32 IP address: ");
    Serial.println(WiFi.localIP());

    // ------------------------------------------------
    // API Routes
    // ------------------------------------------------

    server.on(
        "/unlock",
        HTTP_POST,
        handleUnlock
    );

    server.on(
        "/lock",
        HTTP_POST,
        handleLock
    );

    server.on(
        "/status",
        HTTP_GET,
        handleStatus
    );

    server.onNotFound(
        handleNotFound
    );

    // ------------------------------------------------
    // Start server
    // ------------------------------------------------

    server.begin();

    Serial.println("HTTP server started");
}


// ==================================================
// Main Loop
// ==================================================

void loop() {

    server.handleClient();
}

