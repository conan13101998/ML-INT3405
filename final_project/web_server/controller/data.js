const express = require('express');
const bodyParser = require('body-parser');
const db = require('../config/database');
const json = require("express").json;

const app = express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));

let getHistoryByGender = (req, res) => {
    let queryDB = "SELECT gender, COUNT(*) as count FROM User_info GROUP BY gender ORDER BY count DESC";

    db.getConnection((err, conn) => {
        if(err) {
            res.json({
                success: false,
                reason: err
            })
        }
        else {
            conn.query(queryDB, (err, data) => {
                conn.release();
                if(err) {
                    res.json({
                        success: false,
                        reason: err
                    })
                }
                else {
                    let male_cnt = 0, female_cnt = 0;
                    data.forEach(row => {
                        if(row.gender === 1) {
                            male_cnt = row.count;
                        }
                        else
                            female_cnt = row.count
                    })
                    res.json({
                        success: true,
                        data: {
                            male_cnt: male_cnt,
                            female_cnt: female_cnt
                        }
                    })
                }
            })
        }
    })
}

let getHistoryByAge = (req, res) => {
    let queryDB = "SELECT age, COUNT(*) as count FROM User_info GROUP BY age ORDER BY age ASC";

    db.getConnection((err, conn) => {
        if(err) {
            res.json({
                success: false,
                reason: err
            })
        }
        else {
            conn.query(queryDB, (err, data) => {
                conn.release();
                if(err) {
                    res.json({
                        success: false,
                        reason: err
                    })
                }
                else {
                    let age_cnt = [], age = 0, idx = 0;

                    for(age = 0; age <= 100; age++)
                    {
                        if(idx < data.length && data[idx].age === age) {
                            age_cnt.push(data[idx].count);
                            idx++;
                        }
                        else {
                            age_cnt.push(0)
                        }
                    }
                    res.json({
                        success: true,
                        data: age_cnt
                    })
                }
            })
        }
    })
}

/*
    start_time, end_time format YYYY-MM-DD
* */
let getHistoryByDay = (req, res) => {
    let start_date = req.params.start_date || req.query.start_date;
    let end_date = req.params.end_date || req.query.end_date;

    console.log(start_date);
    console.log(end_date);

    // start_date = start_date.toISOString().replace(/T/, ' ').replace(/\..+/, '').split(' ')[0];
    // end_date = end_date.toISOString().replace(/T/, ' ').replace(/\..+/, '').split(' ')[0];

    let queryDB = "SELECT h.user_id, u.gender, u.age, COUNT(h.user_id) as count\n" +
        "FROM History h, User_info u \n" +
        "WHERE h.user_id = u.user_id AND timestamp(h.time) >= ? AND timestamp(h.time) <= ?\n" +
        "GROUP BY h.user_id"

    db.getConnection((err, conn) => {
        if(err) {
            res.json({
                success: false,
                reason: err
            })
        }
        else {
            conn.query(queryDB, [start_date, end_date], (err, data) => {
                conn.release();
                if(err) {
                    res.json({
                        success: false,
                        reason: err
                    })
                }
                else {
                    let total = 0, gender_cnt = [0, 0], age_cnt = [];
                    for(let i=0; i<=100; i++) {
                        age_cnt.push(0);
                    }
                    data.forEach(row => {
                        total += row.count;
                        gender_cnt[row.gender] += row.count;
                        age_cnt[row.age] += row.count;
                    })
                    res.json({
                        success: true,
                        data: {
                            total: total,
                            by_gender: gender_cnt,
                            by_age: age_cnt
                        }
                    })
                }
            })
        }
    })
}

exports.getHistoryByGender = getHistoryByGender;
exports.getHistoryByAge = getHistoryByAge;
exports.getHistoryByDay = getHistoryByDay;