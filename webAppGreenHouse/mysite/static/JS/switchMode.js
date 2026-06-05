let darkmode = localStorage.getItem("darkmode");

if (darkmode === "active") {
    document.body.classList.add("darkmode");
}

document.addEventListener("DOMContentLoaded", () => {
    const themeSwitch = document.getElementById("theme-switch");
    if (!themeSwitch) return;

    const enableDarkMode = () => {
        document.body.classList.add("darkmode");
        localStorage.setItem("darkmode", "active");
    };

    const disableDarkMode = () => {
        document.body.classList.remove("darkmode");
        localStorage.setItem("darkmode", "inactive"); // Changed this line
    };

    // Check if dark mode was previously enabled
    if (darkmode === "active") {
        enableDarkMode();
      }

    themeSwitch.addEventListener("click", () => {
        darkmode = localStorage.getItem("darkmode");
        if (darkmode !== "active") {
            enableDarkMode();
        } else {
            disableDarkMode();
        }
    });
});
