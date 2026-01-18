/**
 * AIDA Chatbot - LLM Flight Command Interface
 * Connects to LLM server via WebSocket for natural language flight commands
 */

class AIDAChatbot {
  constructor() {
    this.ws = null;
    this.connected = false;
    this.callsign = "AIDA-1";
    // Use same host as page to support remote access (e.g., via Tailscale)
    const wsHost = window.location.hostname || 'localhost';
    this.serverUrl = `ws://${wsHost}:8766`;
    this.messageHistory = [];
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;

    this.init();
  }

  init() {
    this.createUI();
    this.bindEvents();
    this.connect();

    // Welcome message
    setTimeout(() => {
      this.addMessage("pilot", `${this.callsign} autopilot standing by. Ready to accept commands.`);
    }, 500);
  }

  createUI() {
    const container = document.createElement('div');
    container.id = 'chatbot-container';
    container.innerHTML = `
      <div id="chatbot-header">
        <h3>Flight Command</h3>
        <span id="chatbot-status">Connecting...</span>
        <button id="chatbot-toggle">▼</button>
      </div>
      <div id="chatbot-quick-cmds">
        <button class="quick-cmd" data-cmd="status">Status</button>
        <button class="quick-cmd" data-cmd="heading 270">HDG 270</button>
        <button class="quick-cmd" data-cmd="altitude 7000">ALT 7000</button>
        <button class="quick-cmd" data-cmd="land at KHUT">Land KHUT</button>
      </div>
      <div id="chatbot-messages"></div>
      <div id="chatbot-input-container">
        <input type="text" id="chatbot-input" placeholder="Enter command (e.g., 'turn to heading 270')..." />
        <button id="chatbot-send">Send</button>
      </div>
    `;
    document.body.appendChild(container);

    // Store references
    this.container = container;
    this.messagesEl = document.getElementById('chatbot-messages');
    this.inputEl = document.getElementById('chatbot-input');
    this.sendBtn = document.getElementById('chatbot-send');
    this.statusEl = document.getElementById('chatbot-status');
    this.toggleBtn = document.getElementById('chatbot-toggle');
    this.headerEl = document.getElementById('chatbot-header');
  }

  bindEvents() {
    // Send button
    this.sendBtn.addEventListener('click', () => this.sendCommand());

    // Enter key
    this.inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendCommand();
      }
    });

    // Toggle minimize
    this.headerEl.addEventListener('click', (e) => {
      if (e.target.id !== 'chatbot-toggle') return;
      this.container.classList.toggle('minimized');
    });

    // Quick commands
    document.querySelectorAll('.quick-cmd').forEach(btn => {
      btn.addEventListener('click', () => {
        const cmd = btn.dataset.cmd;
        this.inputEl.value = cmd;
        this.sendCommand();
      });
    });
  }

  connect() {
    try {
      this.ws = new WebSocket(this.serverUrl);

      this.ws.onopen = () => {
        this.connected = true;
        this.reconnectAttempts = 0;
        this.setStatus('connected', 'Connected');
        console.log('[Chatbot] Connected to LLM server');
      };

      this.ws.onclose = () => {
        this.connected = false;
        this.setStatus('error', 'Disconnected');
        console.log('[Chatbot] Disconnected from LLM server');
        this.scheduleReconnect();
      };

      this.ws.onerror = (err) => {
        console.error('[Chatbot] WebSocket error:', err);
        this.setStatus('error', 'Error');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleResponse(data);
        } catch (e) {
          console.error('[Chatbot] Failed to parse response:', e);
        }
      };
    } catch (e) {
      console.error('[Chatbot] Failed to connect:', e);
      this.setStatus('error', 'Connection Failed');
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.addMessage("system", "Unable to connect to flight command server. Please check if the LLM server is running.");
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

    setTimeout(() => {
      console.log(`[Chatbot] Reconnecting (attempt ${this.reconnectAttempts})...`);
      this.setStatus('', 'Reconnecting...');
      this.connect();
    }, delay);
  }

  setStatus(state, text) {
    this.statusEl.textContent = text;
    this.statusEl.className = state;
  }

  sendCommand() {
    const text = this.inputEl.value.trim();
    if (!text) return;

    // Add user message to UI
    this.addMessage("user", text);
    this.inputEl.value = '';

    // Send to server
    if (this.connected && this.ws) {
      this.ws.send(JSON.stringify({
        type: "command",
        text: text,
        timestamp: Date.now()
      }));
      this.sendBtn.disabled = true;
      setTimeout(() => { this.sendBtn.disabled = false; }, 1000);
    } else {
      this.addMessage("error", "Not connected to flight command server");
    }
  }

  handleResponse(data) {
    if (data.type === "response") {
      // Pilot response
      this.addMessage("pilot", data.message);

      // If there's an action confirmation, forward to telemetry WebSocket
      if (data.action && data.action.success) {
        this.forwardOverrideToSim(data.action);
        console.log('[Chatbot] Action executed:', data.action);
      }
    } else if (data.type === "override") {
      // Override command from server - forward to telemetry
      this.forwardOverrideToSim(data.data);
    } else if (data.type === "status") {
      // Status update
      this.addMessage("system", data.message);
    } else if (data.type === "error") {
      this.addMessage("error", data.message);
    }
  }

  forwardOverrideToSim(action) {
    // Forward override command to the main telemetry WebSocket (port 8765)
    if (window.wsClient && window.wsClient.readyState === WebSocket.OPEN) {
      let overrideMsg;

      // Handle response format: {success, action, value, target}
      if (action.action !== undefined) {
        // Map LLM action names to controller action names
        let mappedAction = action.action;
        if (action.action === "altitude") mappedAction = "set_altitude";
        if (action.action === "heading") mappedAction = "set_heading";

        overrideMsg = {
          type: "override",
          action: mappedAction,
          value: action.value,
          target: action.target
        };
      }
      // Handle broadcast format: {active, heading, altitude, land_target}
      else if (action.active) {
        // Convert broadcast format to action format
        if (action.altitude !== null) {
          overrideMsg = {
            type: "override",
            action: "set_altitude",
            value: action.altitude
          };
        } else if (action.heading !== null) {
          overrideMsg = {
            type: "override",
            action: "set_heading",
            value: action.heading
          };
        } else if (action.land_target !== null) {
          overrideMsg = {
            type: "override",
            action: "land",
            target: action.land_target
          };
        } else {
          // No actual override values, skip
          return;
        }
      } else {
        // No valid override, skip
        return;
      }

      window.wsClient.send(JSON.stringify(overrideMsg));
      console.log('[Chatbot] Forwarded override to sim:', overrideMsg);
    } else {
      console.warn('[Chatbot] Telemetry WebSocket not connected, cannot forward override');
    }
  }

  addMessage(type, text) {
    const msg = document.createElement('div');
    msg.className = `chat-message ${type}`;

    if (type === "pilot") {
      msg.innerHTML = `<span class="callsign">${this.callsign}</span>${text}`;
    } else {
      msg.textContent = text;
    }

    this.messagesEl.appendChild(msg);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;

    // Store in history
    this.messageHistory.push({ type, text, timestamp: Date.now() });
  }

  // Public method to receive flight state updates
  updateFlightState(state) {
    // Can be called from main viewer to sync state
    this.currentState = state;
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  // Load CSS
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = 'chatbot.css';
  document.head.appendChild(link);

  // Initialize chatbot
  window.aidaChatbot = new AIDAChatbot();
});
