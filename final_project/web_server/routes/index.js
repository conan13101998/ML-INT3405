let express = require('express');
let router = express.Router();
const data = require('../controller/data')

/* GET home page. */
router.get('/', function(req, res, next) {
    res.render('index');
});

router.get('/api/get-history-by-gender', data.getHistoryByGender);

router.get('/api/get-history-by-age', data.getHistoryByAge);

router.get('/api/get-history-by-day', data.getHistoryByDay);

// router.get('/get-history-by-hour/:date', data.getHistoryByHour);

module.exports = router;
