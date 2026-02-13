document.addEventListener('DOMContentLoaded', () => {
    const statusText = document.getElementById('status-text');
    const eventSource = new EventSource('/status');

    eventSource.onmessage = function (event) {
        try {
            const data = JSON.parse(event.data);

            statusText.innerText = data.status;

            if (data.active) {
                statusText.className = 'status active';
            } else {
                statusText.className = 'status idle';
            }
        } catch (e) {
            console.error("Error parsing SSE data:", e);
        }
    };

    eventSource.onerror = function (err) {
        console.error("EventSource failed:", err);
        statusText.innerText = "Connection lost (Reconnecting...)";
        statusText.className = 'status idle';
    };
});