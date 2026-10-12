/* global $argument, $done, $httpClient */

var LOGIN_URL = 'https://console.po0.io/login';
var API_URL = 'https://console.po0.io/modules/servers/penguin/api/productInfo.php';
var UNITS = ['KB', 'MB', 'GB', 'TB', 'PB'];

function decodeArgument(value) {
  try {
    return decodeURIComponent(String(value).replace(/\+/g, ' '));
  } catch (error) {
    return String(value);
  }
}

function parseArguments(argument) {
  var result = {};
  String(argument || '').split('&').forEach(function (part) {
    var separator = part.indexOf('=');
    if (separator < 0) return;
    var key = part.slice(0, separator);
    result[key] = decodeArgument(part.slice(separator + 1));
  });
  if (!result.email || !result.password || !result.service_id ||
      /^\{\{\{.*\}\}\}$/.test(result.email) ||
      /^\{\{\{.*\}\}\}$/.test(result.password) ||
      /^\{\{\{.*\}\}\}$/.test(result.service_id)) {
    throw new Error('请在模块参数中填写 email、password 和 service_id');
  }
  return {
    email: result.email,
    password: result.password,
    serviceId: result.service_id,
  };
}

function decimal(value, minimumFractionDigits, maximumFractionDigits) {
  var factor = Math.pow(10, maximumFractionDigits);
  var rounded = Math.round((value + 1e-12) * factor) / factor;
  var text = rounded.toFixed(maximumFractionDigits);
  if (maximumFractionDigits > minimumFractionDigits && text.indexOf('.') >= 0) {
    text = text.replace(/0+$/, '').replace(/\.$/, '');
  }
  if (text.indexOf('.') < 0 && minimumFractionDigits > 0) {
    text += '.' + new Array(minimumFractionDigits + 1).join('0');
  }
  return text;
}

function formatKB(value) {
  if (typeof value !== 'number' || !isFinite(value) || value <= 0) {
    return { value: '0', unit: 'KB' };
  }
  var unitIndex = 0;
  var scaled = value;
  while (scaled >= 1024 && unitIndex < UNITS.length - 1) {
    scaled /= 1024;
    unitIndex += 1;
  }
  var maximumFractionDigits = scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
  var minimumFractionDigits = scaled < 10 ? 1 : 0;
  return {
    value: decimal(scaled, minimumFractionDigits, maximumFractionDigits),
    unit: UNITS[unitIndex],
  };
}

function readNumber(object, key) {
  var value = object[key];
  if (typeof value !== 'number' || !isFinite(value) || value < 0) {
    throw new Error('登录或接口失败');
  }
  return value;
}

function parseUsage(body) {
  var object;
  try {
    object = typeof body === 'string' ? JSON.parse(body) : body;
  } catch (error) {
    throw new Error('登录或接口失败');
  }
  if (!object || typeof object !== 'object' || object.error || object.message) {
    throw new Error('登录或接口失败');
  }
  return {
    totalTransfer: readNumber(object, 'totalTransfer'),
    remainingTransfer: readNumber(object, 'remainingTransfer'),
    txTransfer: readNumber(object, 'txTransfer'),
    rxTransfer: readNumber(object, 'rxTransfer'),
  };
}

function renderPanel(usage) {
  var remaining = formatKB(usage.remainingTransfer);
  var upload = formatKB(usage.txTransfer);
  var download = formatKB(usage.rxTransfer);
  var ratio = usage.totalTransfer > 0 ? usage.remainingTransfer / usage.totalTransfer : 0;
  var style = usage.remainingTransfer <= 0 ? 'error' : ratio <= 0.1 ? 'alert' : 'good';
  return {
    title: 'Po0GZ 流量',
    content: [
      '上行：' + upload.value + ' ' + upload.unit,
      '下行：' + download.value + ' ' + download.unit,
      '剩余：' + remaining.value + ' ' + remaining.unit,
    ].join('\n'),
    style: style,
    icon: 'chart.bar.fill',
    'icon-color': style === 'good' ? '#22c55e' : style === 'alert' ? '#f59e0b' : '#ef4444',
  };
}

function parseLoginToken(body) {
  var match = String(body || '').match(/name=["']token["'][^>]*value=["']([^"']+)["']/i);
  if (!match) {
    match = String(body || '').match(/value=["']([^"']+)["'][^>]*name=["']token["']/i);
  }
  if (!match || !match[1]) throw new Error('登录或接口失败');
  return match[1];
}

function formEncode(value) {
  return encodeURIComponent(String(value)).replace(/%20/g, '+');
}

function panelError(message) {
  return {
    title: 'Po0GZ 流量',
    content: '读取失败：' + message,
    style: 'error',
    icon: 'exclamationmark.triangle.fill',
    'icon-color': '#ef4444',
  };
}

function requestUsage(config, finish) {
  $httpClient.get({
    url: API_URL + '?serviceId=' + encodeURIComponent(config.serviceId),
    timeout: 10,
    'auto-cookie': true,
    headers: { Accept: 'application/json' },
  }, function (error, response, body) {
    if (error || !response || response.status < 200 || response.status >= 400) {
      finish(panelError('登录或接口失败'));
      return;
    }
    try {
      finish(renderPanel(parseUsage(body)));
    } catch (parseError) {
      finish(panelError(parseError.message));
    }
  });
}

function run(config, finish) {
  $httpClient.get({
    url: LOGIN_URL,
    timeout: 10,
    'auto-cookie': true,
    headers: { Accept: 'text/html' },
  }, function (error, response, body) {
    if (error || !response || response.status < 200 || response.status >= 400) {
      finish(panelError('登录或接口失败'));
      return;
    }
    var token;
    try {
      token = parseLoginToken(body);
    } catch (tokenError) {
      finish(panelError(tokenError.message));
      return;
    }
    $httpClient.post({
      url: LOGIN_URL,
      timeout: 10,
      'auto-cookie': true,
      headers: {
        Accept: 'text/html',
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: [
        'token=' + formEncode(token),
        'username=' + formEncode(config.email),
        'password=' + formEncode(config.password),
      ].join('&'),
    }, function (postError, postResponse) {
      if (postError || !postResponse || postResponse.status < 200 || postResponse.status >= 400) {
        finish(panelError('登录或接口失败'));
        return;
      }
      requestUsage(config, finish);
    });
  });
}

function start() {
  var config;
  try {
    config = parseArguments(typeof $argument === 'string' ? $argument : '');
  } catch (error) {
    $done(panelError(error.message));
    return;
  }
  run(config, function (result) {
    $done(result);
  });
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    formatKB: formatKB,
    parseUsage: parseUsage,
    parseArguments: parseArguments,
    renderPanel: renderPanel,
    parseLoginToken: parseLoginToken,
  };
} else {
  start();
}
