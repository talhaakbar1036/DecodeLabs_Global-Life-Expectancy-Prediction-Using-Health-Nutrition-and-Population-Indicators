document.addEventListener("DOMContentLoaded", function() {
    const buttons = document.querySelectorAll('.animated-button');

    buttons.forEach(button => {
        button.addEventListener('mouseenter', () => {
            button.classList.add('hover');
        });

        button.addEventListener('mouseleave', () => {
            button.classList.remove('hover');
        });

        button.addEventListener('click', () => {
            button.classList.add('active');
            setTimeout(() => {
                button.classList.remove('active');
            }, 300);
        });
    });

    const formInputs = document.querySelectorAll('.form-input');

    formInputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.classList.add('focused');
        });

        input.addEventListener('blur', () => {
            if (input.value === '') {
                input.classList.remove('focused');
            }
        });
    });
});