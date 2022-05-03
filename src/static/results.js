var resultsData;

var AJAXtojson = $.ajax({
    type: 'GET',
    url: '/getresults',
    async: true,
    success: function(response) { 
        resultsData = JSON.parse(JSON.stringify(response));
        console.log(resultsData);

        var table = document.getElementById("table");

        for (var i = 0; i < resultsData["data"].length ; i++){

            var tr = document.createElement("tr");
            var th = document.createElement("th");

            th.innerHTML = i;
            tr.appendChild(th);
            table.appendChild(tr);

            for (var j = 0; j < resultsData["data"][i].length; j++){
                var td = document.createElement("td");
                var string = resultsData["data"][i][j];

                if(j == 0){
                    var date = new Date(string);
                    string = date.toUTCString();
                }

                if(string.toString().includes("?") || string.toString().includes("None")){
                    string = " ";
                }

                td.innerHTML = string;
                tr.appendChild(td);
            }
        }
    },
    contentType: "application/json",
});