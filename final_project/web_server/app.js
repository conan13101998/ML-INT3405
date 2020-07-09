let createError = require('http-errors');
let express = require('express');
let path = require('path');
let cookieParser = require('cookie-parser');
let logger = require('morgan');
const http = require('http');
const https = require('https');
const fs = require('fs');

let indexRouter = require('./routes/index');
let usersRouter = require('./routes/users');

let app = express();

require('dotenv').config();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(express.static(path.join(__dirname, 'public')));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/users', usersRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
    next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
    // set locals, only providing error in development
    res.locals.message = err.message;
    res.locals.error = req.app.get('env') === 'development' ? err : {};

    // render the error page
    res.status(err.status || 500);
    res.render('error');
});

if(process.env.NODE_ENV === 'production'){
    /**
     * point to SSL Cert
     */
    const ssl_key = fs.readFileSync('/etc/letsencrypt/live/sv-procon.uet.vnu.edu.vn/privkey.pem');
    const ssl_cert = fs.readFileSync('/etc/letsencrypt/live/sv-procon.uet.vnu.edu.vn/cert.pem');
    const ca = fs.readFileSync('/etc/letsencrypt/live/sv-procon.uet.vnu.edu.vn/chain.pem');

    let httpsSer = {
        key: ssl_key,
        cert: ssl_cert,
        ca: ca
    };

    https.createServer(httpsSer, app).listen('3003');
}
else if(process.env.NODE_ENV === 'development'){
    http.createServer(app).listen('3000');
}
