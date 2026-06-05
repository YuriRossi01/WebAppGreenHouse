// import:
import {updateTemperatures,parsefloatGlobalData,updateSliderMax,playSlider } from 'constFunction';

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
//giornata / sensibilitè + 1 * numero di valori che mi  arrivano
var legend = L.control({ position: "bottomleft" });

legend.onAdd = function (mymap) {
  var div = L.DomUtil.create("div", "legend");
  div.innerHTML += "<h4>Legend</h4>";
  div.innerHTML += '<i style="background:red"></i><span>>= 30°C</span><br>';
  div.innerHTML += '<i style="background:orange"></i><span>25°C >°C >=30°C</span><br>';
  div.innerHTML += '<i style="background:#FFBF00"></i><span>20°C >°C >=25°C</span><br>';
  div.innerHTML += '<i style="background:yellow"></i><span>20°C > °C >=20°C</span><br>';
  div.innerHTML += '<i style="background:lime"></i><span>10°C > °C >=15°C</span><br>';
  div.innerHTML += '<i style="background:aqua"></i><span>5°C > °C >=10°C</span><br>';
  div.innerHTML += '<i style="background:blue"></i><span>0°C > °C >=5°C</span><br>';
  div.innerHTML += '<i style="background:#000080"></i><span><= 0°C</span><br>';

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
var globalData = [];
//var temperatureReq = [];
//var coldGradient = [1, 1, 1, 1, 1, 1, 1, 1, 1];
let globalTime = '00:00';
// section button reset:
const resetButton = document.getElementById("reset"); // id reset map button
function resetBtn() {
    // Reset della mappa
    mymap.setView([45.402866, 10.998162], 19);

    // Reset dei campi input
    document.getElementById("chooseData").value = "";
    document.getElementById("chooseDataEnd").value = "";
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
const slider = document.getElementById("myRange"); // id slider
const valueSlider = document.getElementById("value"); // id value of slider

var changeValue = slider.addEventListener("input", changeSlider);

function setupSliderRange(dataInizio, dataFine) {

    let inizio = new Date(dataInizio);
    let fine = new Date(dataFine);

    // Calcola la differenza in millisecondi e converti in giorni
    let diffInTime = fine.getTime() - inizio.getTime();
    let diffInDays = Math.round(diffInTime / (1000 * 3600 * 24));

    if(slider){
        slider.value = 0;
        slider.min = 0;
        slider.max = diffInDays; // Lo slider avrà step da 1 giorno
        slider.setAttribute("data-max", diffInDays);
    }


}
function changeSlider() {
  let dataStart = document.getElementById("chooseData").value;
  const sliderValue = parseInt(this.value); // value of slider
  valueSlider.innerText = sliderValue;

  // ora predizione + x volte della sensibilità
  updateTimeIstantDateLabel(dataStart,sliderValue);

  updateTemperatures(mymap,globalData,sliderValue,temperature,heatColdPoints, heatHotPoints);
}

function updateTimeIstantDateLabel(dataStart,sliderValue) {
    let timeIstantDate = document.getElementById("timeIstantDate"); // label in SliderContent
    let currentDate = new Date(dataStart);
    currentDate.setHours(0,0,0,0);
    currentDate.setDate(currentDate.getDate() + sliderValue);

    let day = String(currentDate.getDate()).padStart(2, '0');
    let month = String(currentDate.getMonth() + 1).padStart(2, '0');
    let year = currentDate.getFullYear();

    timeIstantDate.innerHTML = "Data: " + day +"-"+ month +"-"+ year +" (Ore 00:00)";
}

// section Buton autoplay:
let playButton = document.getElementById("playButton");
playButton.addEventListener("click", function() {
    playSlider(playButton, mymap, globalData, temperature, heatColdPoints, heatHotPoints);
});

document.getElementById("sliderSpeed").addEventListener("change", function() {
    if (playButton.textContent === "Pause") {
        // Se sta andando, lo fermiamo e lo facciamo ripartire con la nuova velocità
        playSlider(playButton, mymap, globalData, temperature, heatColdPoints, heatHotPoints); // Ferma
        playSlider(playButton, mymap, globalData, temperature, heatColdPoints, heatHotPoints); // Ricarica
    }
});

// limitation Data to today and pass
function limitationDate() {
  const dateInput = document.getElementById("chooseData");
  const today = new Date().toISOString().split("T")[0];
  dateInput.setAttribute("min", today);
}

// disable DataEnd and Time untill beginning dataInput has a value
let dateInput = document.getElementById("chooseData");
let dateInputEnd = document.getElementById("chooseDataEnd");
dateInput.addEventListener("change", function () {
  if (dateInput.value){
    dateInputEnd.disabled = false;
  }
  else {
    dateInputEnd.disabled = true;
  }
});
const changeMap = document.getElementById("updateMapBtn")
function chooseData() {
  updateMapBtn.disabled = true;
  let loadingOverlay = document.getElementById("loading-overlay");
  loadingOverlay.style.display = "block"; // Mostra la GIF di caricamento
  let arrAlert = [0, 0, 0, 0, 0, 0, 0];
  let dateInput = document.getElementById("chooseData").value;  // Prende il valore in formato YYYY-MM-DD
  let dateInputEnd = document.getElementById("chooseDataEnd").value;
  let chooseBox = document.getElementById("model").value;  // Prende il valore selezionato
  let checkBox = document.getElementById("typeMeanCheckBox");
  let sensibility = document.getElementById("sensibility").value;
  let strMean = document.getElementById("intMean").value;
  let typeOfRange = document.getElementById("typeMean");
  let sensibilityValue = HoursToMinute(document.getElementById("sensibility").value); // take value of sensibility and convert the value to minutes
  let intMeanValue = HoursToMinute(document.getElementById("intMean").value); // take value of ogni tot media and convert it to minutes
  let dimJump = parseInt(document.getElementById("dimSens").value);

  // Verifica se la data e l'ora sono vuote
  if (dateInput === "") { // Data non selezionate.
    arrAlert[0] = 1;
  } else if (checkBox.checked && sensibilityValue > intMeanValue) { // Sensibilità non può essere maggiore di ogni tot media.
    arrAlert[1] = 1;
  } else if (dateInput > dateInputEnd) { // La data è nel passato.
    arrAlert[2] = 1;
  }else if(dateInput == dateInputEnd){ // Data Iniziale non deve essere uguale a quella finale
    arrAlert[3] = 1;
  }

  let errorMessage = generateAlert(arrAlert);
  if (errorMessage) {
    loadingOverlay.style.display = "none"; // Nasconde la GIF
    alert(errorMessage);
    updateMapBtn.disabled = false;
  }
  else {
    // Crea l'URL della richiesta
    const xhr = new XMLHttpRequest();
    let url = "http://localhost:8000/reqTempMonthDay/?salto=" + sensibility + "&dataInizio=" + encodeURIComponent(dateInput) + "&dataFine=" + encodeURIComponent(dateInputEnd) + "&model=" + chooseBox;
    if (checkBox.checked) {
      if (typeOfRange == "fixed")
        dimJump = Math.floor((intMeanValue / sensibilityValue) / 2);
      url = "http://localhost:8000/reqTempMonthDayMedia/?precisioneSalto=" + sensibility + "&ogniTotMedia=" + strMean + "&dataInizio=" + encodeURIComponent(dateInput) + "&dataFine=" + encodeURIComponent(dateInputEnd) + "&model=" + chooseBox + "&dimIntervallo=" + dimJump;
    }

    xhr.open('GET', url);
    xhr.responseType = "json";
    xhr.onload = function () {
      if (xhr.readyState == 4 && xhr.status == 200) {
        // Aggiorna i dati dei sensori se la richiesta ha successo0
        loadingOverlay.style.display = "none"; // Nasconde la GIF
        let data = xhr.response;
        globalData = [];
        globalData = parsefloatGlobalData(data,globalData);
        updateTemperatures(mymap,globalData,0,temperature, heatColdPoints, heatHotPoints);
        setupSliderRange(dateInput,dateInputEnd);
        updateMapBtn.disabled = false;
      } else { // Errore recupero Dati.
        arrAlert[4] = 1;
        generateAlert(arrAlert);
        updateMapBtn.disabled = false;
        loadingOverlay.style.display = "none"; // Nasconde la GIF in caso di errore
      }
    };
    xhr.onerror = function () { // Errore nella richiesta.
      loadingOverlay.style.display = "none"; // Nasconde la GIF in caso di errore
      arrAlert[5] = 1;
      generateAlert(arrAlert);
    };
    // Invia la richiesta
    xhr.send();
  }
}
changeMap.addEventListener("click",chooseData)

function generateAlert(arrAlert) {
  let errorMessages = [];
  let errorMessagesList = [
    'Data non selezionate.',
    'Sensibilità non può essere maggiore di ogni tot media.',
    'La data è nel passato.',
    'Data Iniziale non deve essere uguale a quella finale',
    'Errore recupero Dati.',
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
  let dateInput = document.getElementById('chooseData'); // Prende l'elemento di input data
  let dateInputEnd = document.getElementById("chooseDataEnd");
  let selectedDate = new Date(dateInput.value); // Converte la data selezionata in un oggetto Date
  let minDateEnd = new Date(selectedDate);
  let maxDateEnd = new Date(selectedDate);
  let now = new Date();                         // Ottiene la data e ora attuali
  let isToday = selectedDate.toDateString() === now.toDateString(); // Verifica se la data selezionata è oggi
  let minHours;

  minDateEnd.setDate(minDateEnd.getDate() + 1); // next Day
  maxDateEnd.setMonth(maxDateEnd.getMonth() + 1); // next Mounth

  // Set the limits on the end date input
  dateInputEnd.min = minDateEnd.toISOString().split("T")[0]; // Format YYYY-MM-DD
  dateInputEnd.max = maxDateEnd.toISOString().split("T")[0]; // Format YYYY-MM-DD

  if(!dateInputEnd.value){
    dateInputEnd.value = dateInputEnd.min;
  }

  if (isToday) {
    minHours = String(now.getHours() + 1, 0, 0, 0).padStart(2, '0');
  }
  else { minHours = '00' }
  let minTime = minHours + ':00';
  globalTime = minTime;

}
dateInput.addEventListener("change", limitationTime);

// se il valore ora scelta è < di global time allora value = globalTime else nulla
//playButton.addEventListener("click", playSlider);
let upDimRange = document.getElementById("sensibility");
let upIntMean = document.getElementById("intMean");
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
function hideDimJump(){
    const typeMeanCheckBox = document.getElementById("typeMeanCheckBox");
    const typeMeanContainer = document.getElementById("typeMeanContainer");
    const dimSensibilityContainer = document.getElementById("dimSensibilityContainer");
    const typeMeanSelect = document.getElementById("typeMean");

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
}
document.addEventListener("DOMContentLoaded", hideDimJump());

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
