export const commonRes = (data: Object) => {
    return {
        code: 200,
        data: data,
        message: 'success'
    }
}

export const commonError = (code: number, message: string) => {
    return {
        code: code,
        data: null,
        message: message || '未知错误'
    }
}