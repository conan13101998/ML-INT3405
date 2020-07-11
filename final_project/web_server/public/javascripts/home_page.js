$(document).ready(() => {
    home.getHistoryByAge();
    $("#genderChartContainer").hide();
    $("#dayChartContainer").hide();
    $("#ageChartContainer").show();
})

$("#ageChartBtn").on('click', () => {
    home.getHistoryByAge();
    $("#genderChartContainer").hide();
    $("#dayChartContainer").hide();
    $("#ageChartContainer").show();
})

$("#genderChartBtn").on('click', () => {
    home.getHistoryByGender();
    $("#genderChartContainer").show();
    $("#dayChartContainer").hide();
    $("#ageChartContainer").hide();
})

$("#dayChartBtn").on('click', () => {
    let options = {
        format: 'yyyy-mm-dd'
    }
    let start_date_ele = $("#startDatePicker");
    let end_date_ele = $("#endDatePicker");
    start_date_ele.val("");
    end_date_ele.val("");
    let start_date = M.Datepicker.init(start_date_ele, options);
    let end_date = M.Datepicker.init(end_date_ele, options);

    home.getHistoryByDay();
    $("#genderChartContainer").hide();
    $("#dayChartContainer").show();
    $("#ageChartContainer").hide();
})

$("#getDayDataBtn").on('click', () => {
    let start_date = $("#startDatePicker").val();
    let end_date = $("#endDatePicker").val();
    console.log(start_date);
    console.log(end_date);
    home.getHistoryByDay(start_date, end_date);
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
            data: {
                start_date: start_date,
                end_date: end_date
            },
            success: res => {
                console.log(res);
                $("#totalCustomer").innerText = "Total customer: " + res.data.total;

                let age_chart = am4core.create("dayChartAge", am4charts.PieChart);
                let age_data = [];
                for(let i=0; i<res.data.by_age.length; i++) {
                    if(res.data.by_age[i] !== 0)
                        age_data.push({
                            age: i,
                            count: res.data.by_age[i]
                        })
                }
                console.log(age_data);
                age_chart.data = age_data;
                let pieSeries1 = age_chart.series.push(new am4charts.PieSeries());
                pieSeries1.dataFields.value = "count";
                pieSeries1.dataFields.category = "age";
                age_chart.legend = new am4charts.Legend();

                let gender_chart = am4core.create("dayChartGender", am4charts.PieChart);
                let gender_data = [];
                gender_data.push({
                    gender: "Male",
                    count: res.data.by_gender[1]
                });
                gender_data.push({
                    gender: "Female",
                    count: res.data.by_gender[0]
                })
                console.log(gender_data);
                gender_chart.data = gender_data;
                let pieSeries2 = gender_chart.series.push(new am4charts.PieSeries());
                pieSeries2.dataFields.value = "count";
                pieSeries2.dataFields.category = "gender";
                gender_chart.legend = new am4charts.Legend();

            },
            error: (e, h, r) => {
                M.toast({html: e.responseText, classes: 'danger'});
            }
        })
    }
})