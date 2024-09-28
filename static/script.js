$(document).ready(function() {
    $('#submitBtn').on('click', function() {
        const text = $('#inputText').val();

        if (text.trim() === '') {
            alert('Please enter some text!');
            return;
        }

        // Show the loader
        $('#loader').show();

        $.ajax({
            url: '/predict',  // Adjust this URL based on your Flask route
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ text: text }),
            success: function(response) {
                $('#output').html(`
                    <strong>Input:</strong> ${text}<br>
                    <strong>Prediction:</strong> ${response.prediction}
                `);
                $('#output').show(); // Show the output div
            },
            error: function() {
                alert('Error occurred while making the prediction.');
            },
            complete: function() {
                // Hide the loader after the request is complete
                $('#loader').hide();
            }
        });
    });

    $('#clearBtn').on('click', function() {
        $('#inputText').val(''); // Clear the input field
        $('#output').hide(); // Hide the output div
        $('#output').html(''); // Clear the output content
    });
});
