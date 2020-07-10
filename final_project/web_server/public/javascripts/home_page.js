$(document).ready(() => {
    home.getHistoryByAge();
})

$("#ageChartBtn").on('click', () => {
    home.getHistoryByAge();
    $("#genderChartContainer").classList.add("hide");
    $("#dayChartChartContainer").classList.add("hide");
    $("#ageChartContainer").classList.remove("hide");
})

$("#genderChartBtn").on('click', () => {
    home.getHistoryByGender();
    $("#genderChartContainer").classList.remove("hide");
    $("#dayChartChartContainer").classList.add("hide");
    $("#ageChartContainer").classList.add("hide");
    $("#ageChartContainer").style.display = "block";
})

$("#dayChartBtn").on('click', () => {
    home.getHistoryByDay();
    $("#genderChartContainer").classList.add("hide");
    $("#dayChartChartContainer").classList.remove("hide");
    $("#ageChartContainer").classList.add("hide");
})

let home = Object.create({
    getHistoryByAge: () => {
        $.ajax({
            url: '/api/get-history-by-age',
            method: 'GET',
            success: res => {
                let age_chart = am4core.create("ageChart", am4charts.PieChart);
                let age_data = [];
                for(let i=0; i<res.data.length; i++) {
                    if(res.data[i] !== 0)
                        age_data.push({
                            age: i,
                            count: res.data[i]
                        })
                }
                age_chart.data = age_data;
                let pieSeries = age_chart.series.push(new am4charts.PieSeries());
                pieSeries.dataFields.value = "count";
                pieSeries.dataFields.category = "age";
                age_chart.legend = new am4charts.Legend();
            },
            error: (e, h, r) => {
                M.toast({html: e.responseText, class: 'danger'});
            }
        })
    },
    getHistoryByGender: () => {
        $.ajax({
            url: '/api/get-history-by-gender',
            method: 'GET',
            success: res => {
                let gender_chart = am4core.create("genderChart", am4charts.PieChart);
                let gender_data = [];
                gender_data.push({
                    gender: "Male",
                    count: res.data.male_cnt
                });
                gender_data.push({
                    gender: "Female",
                    count: res.data.female_cnt
                })
                gender_chart.data = gender_data;
                let pieSeries = gender_chart.series.push(new am4charts.PieSeries());
                pieSeries.dataFields.value = "count";
                pieSeries.dataFields.category = "gender";
                gender_chart.legend = new am4charts.Legend();
            },
            error: (e, h, r) => {
                M.toast({html: e.responseText, classes: 'danger'});
            }
        })
    },
    getHistoryByDay: (start_date, end_date) => {
        $.ajax({
            url: '/api/get-history-by-day',
            method: 'GET',
            success: res => {
                M.toast({html: res.data.total, classes: 'green'});
                M.toast({html: res.data.by_gender, classes: 'blue'});
            },
            error: (e, h, r) => {
                M.toast({html: e.responseText, classes: 'danger'});
            }
        })
    }
})