function sendMessage() {
    var userText = document.getElementById('user_input').value;

    // Display user message in the chat box
    var chatBox = document.getElementById('chat-box');
    var userMessage = document.createElement('div');
    userMessage.className = 'user-message';
    userMessage.textContent = userText;
    chatBox.appendChild(userMessage);

    // Clear the input field
    document.getElementById('user_input').value = '';

    // Fetch bot response
    fetch('/get?msg=' + encodeURIComponent(userText))
        .then(response => response.text())
        .then(data => {
            var botMessage = document.createElement('div');
            botMessage.className = 'bot-message';
            botMessage.textContent = data;
            chatBox.appendChild(botMessage);

            // Scroll to the bottom of the chat box
            chatBox.scrollTop = chatBox.scrollHeight;
        });
}
