// import:
import {updateTemperatures,parsefloatGlobalData,updateSliderMax,playSlider,intervalTemp } from 'constFunction';

// Initialize map with coordinates and \zoom
const mymap = L.map("map");
mymap.setView([45.402866, 10.999162], 19);
var tileLayer = L.tileLayer(

    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    //"https://tiles.stadiamaps.com/tiles/osm_bright/{z}/{x}/{y}{r}.png",
    {
        attribution:
        '&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        //ext: 'png'
        maxZoom: 19,
        minZoom: 16,
        maxNativeZoom: 19
    }
);
tileLayer.addTo(mymap);


var legend = L.control({ position: "bottomleft" });

legend.onAdd = function(mymap) {
  var div = L.DomUtil.create("div","legend");
  div.innerHTML += "<h4>Legenda Termica</h4>";

    div.innerHTML += '<i style="background: #ff0000"></i><span>&gt;= 30°C</span><br>';
    div.innerHTML += '<i style="background: #ff7f00"></i><span>25°C a 30°C</span><br>';
    div.innerHTML += '<i style="background: #ffff00"></i><span>20°C a 25°C</span><br>';
    div.innerHTML += '<i style="background: #00ff00"></i><span>15°C a 20°C</span><br>';
    div.innerHTML += '<i style="background: #00ffff"></i><span>10°C a 15°C</span><br>';
    div.innerHTML += '<i style="background: #0000ff"></i><span>0°C a 10°C</span><br>';
    div.innerHTML += '<i style="background: #000080"></i><span>&lt;= 0°C</span><br>';

  return div;
};
legend.addTo(mymap);

// Coordinates of the sensors and their detected temperature
var temperature = [
  [45.402866, 10.998162, -10], // lat, lng, (temperature) intensity
  [45.40265, 10.99828, -11],
  [45.40269, 10.99845, -12],
  [45.4029, 10.99831, -13],
  [45.40286, 10.99841, -14],
  [45.40276, 10.99832, -15],
  [45.40279, 10.99823, -16],
  [45.40279, 10.99823, -17]
];
let globalData = [];

let globalTime = '00:00';

// section button reset:
const resetButton = document.getElementById("reset"); // id reset map button
function resetBtn() {
    // Reset della mappa
    mymap.setView([45.402866, 10.998162], 19);

    // Reset dei campi input
    document.getElementById("chooseData").value = "";
    document.getElementById("time").value = "";
    document.getElementById("model").selectedIndex = 0; // Torna alla prima opzione
}
resetButton.addEventListener("click", resetBtn);

// Initialize heat layers (cold)
var heatColdPoints = L.heatLayer(temperature, {
  radius: 20,
  minOpacity: 0.7,
  gradient: {
    0.0: "red",
    0.14: "orange",
    0.28: "#FFBF00",
    0.42: "yellow",
    0.56: "lime",
    0.7: "aqua",
    0.84: "blue",
    1.0: "black"
  }
});
heatColdPoints.addTo(mymap); // add heatColdlayer in to the map

// Initialize heat layers (Hot)
var heatHotPoints = L.heatLayer(temperature, {
  radius: 20,
  minOpacity: 0.7,
  gradient: {
    0.0: "black",
    0.14: "blue",
    0.28: "aqua",
    0.42: "lime",
    0.56: "yellow",
    0.7: "#FFBF00",
    0.84: "orange",
    1.0: "red"
  }
});
heatHotPoints.addTo(mymap);

// Event listener for the slider
let slider = document.getElementById("myRange"); // id slider
let valueSlider = document.getElementById("value"); // id value of slider

var changeValue = slider.addEventListener("input", changeSlider);

function changeSlider() {
    let timeInput = document.getElementById("time").value;
    let sensibilityValue = HoursToMinute(document.getElementById("sensibility").value);

    const sliderValue = parseInt(this.value); // value of slider
    valueSlider.innerText = sliderValue;

    // ora predizione + x volte della sensibilità
    updateTimeIstantDateLabel(timeInput, sensibilityValue, sliderValue);
    updateTemperatures(mymap,globalData,sliderValue,temperature,heatColdPoints, heatHotPoints);
}

function updateTimeIstantDateLabel(timeInput, sensibilityValue, sliderValue) {
    let data = document.getElementById("chooseData").value;
    let timeIstantDate = document.getElementById("timeIstantDate"); // label in SliderContent
    let time = new Date(data);

    let [hours, minutes] = timeInput.split(":").map(Number);
    time.setHours(hours, minutes, 0, 0);

    let increment = sensibilityValue * sliderValue;
    time.setMinutes(time.getMinutes() + increment);


    let year = time.getFullYear();
    let month = String(time.getMonth() + 1).padStart(2, '0');
    let day = String(time.getDate()).padStart(2, '0');

    let updateDate = `${year}-${month}-${day}`;
    let updateTime = time.toTimeString().split(":").slice(0, 2).join(":");

    timeIstantDate.innerHTML = "Data: " + updateDate + " Ora: " + updateTime;
}

// section Buton autoplay:
let playButton = document.getElementById("playButton");
playButton.addEventListener("click", function() {
    playSlider(playButton, mymap, globalData, temperature, heatColdPoints, heatHotPoints);
});

// Play / Pause Button
document.getElementById("sliderSpeed").addEventListener("change", function() {
    if (playButton.textContent === "Pause") {
        // Se sta andando, lo fermiamo e lo facciamo ripartire con la nuova velocità
        playSlider(playButton, mymap, globalData, temperature, heatColdPoints, heatHotPoints); // Ferma
        playSlider(playButton, mymap, globalData, temperature, heatColdPoints, heatHotPoints); // Ricarica
    }
});

//limitation Data to today and pass
function limitationDate() {
    const dateInput = document.getElementById("chooseData");
    const today = new Date().toISOString().split("T")[0];
    dateInput.setAttribute("min", today);
}

let dateInput = document.getElementById("chooseData");
let time = document.getElementById("time");
if (!dateInput.value){
    time.disabled = true;
    }

dateInput.addEventListener("change", function () {
    if (dateInput.value)
        time.disabled = false;
    else {
        time.disabled = true;
    }
});

const changeMap = document.getElementById("updateMapBtn")
function chooseData() {
    updateMapBtn.disabled = true;
    let loadingOverlay = document.getElementById("loading-overlay");
    loadingOverlay.style.display = "block"; // Mostra la GIF di caricamento
    let arrAlert = [0, 0, 0, 0, 0];
    let dateInput = document.getElementById("chooseData").value;  // Prende il valore in formato YYYY-MM-DD
    let timeInput = document.getElementById("time");  // Prende il valore in formato HH:MM
    let chooseBox = document.getElementById("model").value;  // Prende il valore selezionato
    let checkBox = document.getElementById("typeMeanCheckBox");
    let sensibility = document.getElementById("sensibility").value;
    let strMean = document.getElementById("intMean").value;
    let typeOfRange = document.getElementById("typeMean");
    let sensibilityValue = HoursToMinute(document.getElementById("sensibility").value); // take value of sensibility and convert the value to minutes
    let intMeanValue = HoursToMinute(document.getElementById("intMean").value); // take value of ogni tot media and convert it to minutes
    let dimJump = parseInt(document.getElementById("dimSens").value);


    let now = new Date();
    let selectedDate = new Date(dateInput + 'T' + timeInput);

    let hours = String(now.getHours()).padStart(2, '0');
    let minutes = String(now.getMinutes()).padStart(2, '0');
    let timeNow = hours + ':' + minutes;

    let minTime = '21:00';
    timeInput.min = minTime;

    timeInput.setAttribute("min", minTime);
    timeInput = document.getElementById("time").value;


    // Verifica se la data e l'ora sono vuote
    if (timeInput === "" || dateInput === "") {
        arrAlert[0] = 1;
    } else if (checkBox.checked && sensibilityValue > intMeanValue) {
        arrAlert[1] = 1;
    } else if (dateInput < selectedDate) {
        arrAlert[2] = 1;
    } else if (selectedDate < now) {
        arrAlert[3] = 1;
    }

    let errorMessage = generateAlert(arrAlert);
    if (errorMessage) {
        loadingOverlay.style.display = "none"; // Nasconde la GIF
        alert(errorMessage);

    }
    else {

        // Crea l'URL della richiesta
        const xhr = new XMLHttpRequest();
        let url = "http://localhost:8000/reqTempDay/?salto=" + sensibility + "&day=" + encodeURIComponent(dateInput) + "&time=" + encodeURIComponent(timeInput) + "&model=" + chooseBox;
        if (checkBox.checked) {
        if (typeOfRange == "fixed")
            dimJump = Math.floor((intMeanValue / sensibilityValue) / 2);
          url = "http://localhost:8000/reqTempDayMedia/?precisioneSalto=" + sensibility + "&ogniTotMedia=" + strMean + "&day=" + encodeURIComponent(dateInput) + "&time=" + encodeURIComponent(timeInput) + "&model=" + chooseBox + "&dimIntervallo=" + dimJump;
        }

        xhr.open('GET', url);
        xhr.responseType = "json";
        xhr.onload = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            // Aggiorna i dati dei sensori se la richiesta ha successo
            loadingOverlay.style.display = "none"; // Nasconde la GIF
            let data = xhr.response;
            globalData = [];
            globalData = parsefloatGlobalData(data,globalData);
            updateTemperatures(mymap,globalData,0,temperature, heatColdPoints, heatHotPoints);

        } else {
            arrAlert[5] = 1;
            generateAlert(arrAlert);

            loadingOverlay.style.display = "none"; // Nasconde la GIF in caso di errore
        }
        };
        xhr.onerror = function () {
        loadingOverlay.style.display = "none"; // Nasconde la GIF in caso di errore
        arrAlert[6] = 1;
        generateAlert(arrAlert);
        };
        // Invia la richiesta
        xhr.send();
    }
    updateMapBtn.disabled = false;
}
changeMap.addEventListener("click",chooseData)

// Allert Message Function:
function generateAlert(arrAlert) {
  let errorMessages = [];
  let errorMessagesList = [
    'Data o ora non selezionate.',
    'Sensibilità non può essere maggiore di ogni tot media.',
    'La data è nel passato.',
    'L\'ora selezionata è nel passato.',
    'Errore recupero Dati.',
    'Dimensione salto non selezionato',
    'Errore nella richiesta.',
  ];
  for (let i = 0; i < arrAlert.length; i++) {
    if (arrAlert[i] === 1) {
      errorMessages.push(errorMessagesList[i]);
    }
  }
  return errorMessages.length > 0 ? errorMessages.join('; ') + '.' : '';
}


function limitationTime() {
  const dateInput = document.getElementById('chooseData'); // Prende l'elemento di input data
  let selectedDate = new Date(dateInput.value); // Converte la data selezionata in un oggetto Date
  let now = new Date();                         // Ottiene la data e ora attuali
  let isToday = selectedDate.toDateString() === now.toDateString(); // Verifica se la data selezionata è oggi
  let minHours;
  if (isToday) {
    minHours = String(now.getHours() + 1, 0, 0, 0).padStart(2, '0');
  }
  else { minHours = '00' }
  let minTime = minHours + ':00';
  globalTime = minTime;

}
dateInput.addEventListener("change", limitationTime);

const upCheckTime =document.getElementById('time');

function checkTime() {
  let timeInput = document.getElementById('time');
  if (timeInput.value < globalTime) {
    timeInput.value = globalTime;
  }
}

upCheckTime.addEventListener("change", checkTime);

const upDimRange = document.getElementById("sensibility");
const upIntMean = document.getElementById("intMean");

function dimRange() {
  let checkBox = document.getElementById("typeMeanCheckBox");
  let sensibilityValue = HoursToMinute(document.getElementById("sensibility").value); // take value of sensibility and convert the value to minutes
  let intMeanValue = HoursToMinute(document.getElementById("intMean").value); // take value of ogni tot media and convert it to minutes
  let dimIntervallo = document.getElementById("dimSens");

  // Controlla il valore selezionato nel tipo di media
  // If checkbox is enabled
  if (checkBox.checked) {
    // Clear existing options
    dimIntervallo.innerHTML = '';
    let step = intMeanValue / sensibilityValue;

    for (let i = 0; i <= step; i++) {
      // Corrected the syntax here
      let option = document.createElement("option");
      option.value = i;
      option.text = i;
      dimIntervallo.appendChild(option);
    }
  }
}
upDimRange.addEventListener("change",dimRange);
upIntMean.addEventListener("change",dimRange);


document.addEventListener("DOMContentLoaded", function () {
  const typeMeanCheckBox = document.getElementById("typeMeanCheckBox");
  const typeMeanContainer = document.getElementById("typeMeanContainer");
  const dimSensibilityContainer = document.getElementById("dimSensibilityContainer");
  const typeMeanSelect = document.getElementById("typeMean");

  // Mostra/Nasconde il tipo di media (fisso/dinamico) quando la checkbox viene attivata
  typeMeanCheckBox.addEventListener("change", function () {
    if (typeMeanCheckBox.checked) {
      typeMeanContainer.style.display = "flex"; // Mostra il selettore tipo di media
      typeMeanSelect.value = "fixed"; // Predefinito a "Intervallo fisso"
      dimSensibilityContainer.style.display = "none"; // Nascondi la Dimensione del Salto
    } else {
      typeMeanContainer.style.display = "none"; // Nascondi il tipo di media
      dimSensibilityContainer.style.display = "none"; // Nascondi la Dimensione del Salto
    }
  });
  // Mostra/Nasconde "Dim Intervallo" in base alla scelta nel tipo di media
  typeMeanSelect.addEventListener("change", function () {
    if (typeMeanSelect.value === "dynamic") {
      dimSensibilityContainer.style.display = "flex"; // Mostra "Dim Intervallo"
    } else {
      dimSensibilityContainer.style.display = "none"; // Nascondi "Dim Intervallo"
    }
  });
});

function HoursToMinute(hours) {
  if (hours.endsWith(".m")) return parseInt(hours);
  if (hours.endsWith(".h")) return parseInt(hours) * 60;
  return 0;
}


// ----------- Prints section --------------\\
// Function to update temperature labels
function initializeTable() {
  for (var i = 0; i < temperature.length; i++) {
    var valueTemperature = temperature[i][2];
    document.getElementById("sens" + (i + 1)).innerText = valueTemperature.toFixed(2) + "°C";
  }
}


// Initial setup
initializeTable(); // Display table initially
limitationDate();

