const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');
const interactionLayer = document.getElementById('interactionLayer');
const videoSource = document.getElementById('videoSource');
const startBtn = document.getElementById('startBtn');
const modeSelect = document.getElementById('modeSelect');
const labelSelect = document.getElementById('labelSelect');
const textPrompt = document.getElementById('textPrompt');
const clearBtn = document.getElementById('clearBtn');
const maskThresh = document.getElementById('maskThresh');
const maskThreshVal = document.getElementById('maskThreshVal');

const rgbFps = document.getElementById('rgbFps');
const segFps = document.getElementById('segFps');
const currentMode = document.getElementById('currentMode');
const promptStatus = document.getElementById('promptStatus');

let ws = null;
let isDrawing = false;
let startX, startY;
let currentFrameW = 0;
let currentFrameH = 0;

function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'frame') {
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                
                // Update interaction layer size to match the RGB part (1/3 of width)
                // Assuming [RGB | Overlay | Mask]
                currentFrameW = img.width / 3;
                currentFrameH = img.height;
                
                const rect = canvas.getBoundingClientRect();
                const scaleX = rect.width / canvas.width;
                const scaleY = rect.height / canvas.height;
                
                interactionLayer.style.width = (rect.width / 3) + 'px';
                interactionLayer.style.height = rect.height + 'px';
                // Adjust left position based on canvas alignment
                interactionLayer.style.left = (canvas.offsetLeft) + 'px';
                interactionLayer.style.top = (canvas.offsetTop) + 'px';
            };
            img.src = `data:image/jpeg;base64,${data.image}`;
            
            // Update stats
            rgbFps.innerText = data.stats.rgb_fps;
            segFps.innerText = data.stats.seg_fps;
            currentMode.innerText = data.stats.mode;
            
            const ps = data.stats.prompt_summary;
            promptStatus.innerText = `T: ${ps.text || '-'}, P: ${ps.points}, B: ${ps.boxes}`;
        }
    };

    ws.onclose = () => {
        console.log('WS connection closed');
        setTimeout(connect, 2000);
    };
}

startBtn.onclick = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
        type: 'start',
        payload: { source: videoSource.value }
    }));
};

textPrompt.oninput = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
        type: 'set_text',
        payload: { text: textPrompt.value }
    }));
};

clearBtn.onclick = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'clear_prompts' }));
    textPrompt.value = '';
};

maskThresh.oninput = () => {
    const val = parseFloat(maskThresh.value);
    maskThreshVal.innerText = val;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
        type: 'set_thresholds',
        payload: { mask_threshold: val }
    }));
};

// Interaction Layer Events
interactionLayer.onmousedown = (e) => {
    if (modeSelect.value !== 'interactive') return;
    
    const rect = interactionLayer.getBoundingClientRect();
    startX = (e.clientX - rect.left) / rect.width * currentFrameW;
    startY = (e.clientY - rect.top) / rect.height * currentFrameH;
    isDrawing = true;
};

interactionLayer.onmouseup = (e) => {
    if (!isDrawing) return;
    isDrawing = false;
    
    const rect = interactionLayer.getBoundingClientRect();
    const endX = (e.clientX - rect.left) / rect.width * currentFrameW;
    const endY = (e.clientY - rect.top) / rect.height * currentFrameH;
    
    const dist = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2));
    
    if (dist < 10) {
        // Treat as a point click
        ws.send(JSON.stringify({
            type: 'add_point',
            payload: {
                x: Math.round(startX),
                y: Math.round(startY),
                label: parseInt(labelSelect.value)
            }
        }));
    } else {
        // Treat as a box
        ws.send(JSON.stringify({
            type: 'add_box',
            payload: {
                x1: Math.round(Math.min(startX, endX)),
                y1: Math.round(Math.min(startY, endY)),
                x2: Math.round(Math.max(startX, endX)),
                y2: Math.round(Math.max(startY, endY)),
                label: parseInt(labelSelect.value)
            }
        }));
    }
};

connect();
