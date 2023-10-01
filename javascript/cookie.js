function cookieSet(name, value) {
    try {
        document.cookie = name + "=" + value
    } catch (e) {
        console.warn(`Failed to save ${name} to cookie: ${e}`);
    }
}

function cookieGet(name, def) {
    try {
        let arr, reg = new RegExp("(^| )" + name + "=([^;]*)(;|$)");

        if (arr = document.cookie.match(reg))

            return arr[2];
        else
            return null;
    } catch (e) {
        console.warn(`Failed to load ${name} from cookie: ${e}`);
    }

    return def;
}

function cookieRemove(name) {
    try {
        document.cookie = name + "="
    } catch (e) {
        console.warn(`Failed to remove ${name} from localStorage: ${e}`);
    }
}
