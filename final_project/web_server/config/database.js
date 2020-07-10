const database = require('mysql');

require('dotenv').config();

const db_connection = database.createPool({
    connectionLimit: 1000,
    host: 'sv-procon.uet.vnu.edu.vn',
    user: 'root',
    password: 'iotlab2018',
    database: 'age_gender_prediction'
});

module.exports = db_connection