#include <WiFi.h>
#include <WebServer.h>

// ==================================================
// Wi-Fi Configuration
// ==================================================

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ==================================================
// ESP32 API Authentication
// ==================================================

// Must exactly match ESP32_SHARED_SECRET in FastAPI .env
const char* DOOR_SHARED_SECRET = "hbtCMA2Fs9qNqMZlGAPf8cblk68quxgZe3t2O5-e18g";

// Header used by FastAPI
const char* AUTH_HEADER = "X-Door-Authorization";

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
// Helper: Authentication
// ==================================================

bool isAuthorized() {

    if (!server.hasHeader(AUTH_HEADER)) {

        Serial.println(
            "Unauthorized request: missing authentication header"
        );

        return false;
    }

    String providedSecret =
        server.header(AUTH_HEADER);

    if (!providedSecret.equals(DOOR_SHARED_SECRET)) {

        Serial.println(
            "Unauthorized request: invalid authentication secret"
        );

        return false;
    }

    return true;
}

// ==================================================
// Helper: Unauthorized Response
// ==================================================

void sendUnauthorizedResponse() {

    server.send(
        401,
        "application/json",
        "{\"success\":false,\"message\":\"Unauthorized\"}"
    );
}

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

    if (!isAuthorized()) {

        sendUnauthorizedResponse();

        return;
    }

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

    if (!isAuthorized()) {

        sendUnauthorizedResponse();

        return;
    }

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

    if (!isAuthorized()) {

        sendUnauthorizedResponse();

        return;
    }

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

    // ------------------------------------------------
    // Configure LED
    // ------------------------------------------------

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
    // Collect Authentication Header
    // ------------------------------------------------

    const char* headerKeys[] = {
        AUTH_HEADER
    };

    server.collectHeaders(
        headerKeys,
        1
    );

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
    // Start Server
    // ------------------------------------------------

    server.begin();

    Serial.println("HTTP server started");
    Serial.println("Door API authentication enabled");
}

// ==================================================
// Main Loop
// ==================================================

void loop() {

    server.handleClient();
}