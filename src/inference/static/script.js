document.addEventListener('DOMContentLoaded', () => {
    const statusText = document.getElementById('status-text');
    const interpretationCheckbox = document.getElementById('interpretation-checkbox');
    const interpretationBox = document.getElementById('interpretation-box');
    const interpretationText = document.getElementById('interpretation-text');
    const eventSource = new EventSource('/status');

    // Toggle interprétation
    interpretationCheckbox.addEventListener('change', () => {
        if (interpretationCheckbox.checked) {
            interpretationBox.classList.remove('hidden');
        } else {
            interpretationBox.classList.add('hidden');
        }
    });

    eventSource.onmessage = function (event) {
        try {
            const data = JSON.parse(event.data);

            statusText.innerText = data.status;

            if (data.active) {
                statusText.className = 'status active';
            } else {
                statusText.className = 'status idle';
            }

            // Mise à jour du texte d'interprétation
            if (data.interpretation) {
                interpretationText.innerText = data.interpretation;
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