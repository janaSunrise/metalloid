# Foundation

## What it is

Foundation is Apple's library of everyday Objective-C classes: strings,
arrays, dictionaries, numbers, errors, autorelease pools. Roughly Python's
built-in types and part of its standard library.

| Foundation     | Python                                   |
| -------------- | ---------------------------------------- |
| `NSString`     | `str`                                    |
| `NSArray`      | `list`                                   |
| `NSDictionary` | `dict`                                   |
| `NSNumber`     | `int`, `float`                           |
| `NSError`      | an exception, returned instead of raised |

## Three layers

| Library        | Provides                                                     |
| -------------- | ------------------------------------------------------------ |
| libobjc        | the language: classes, selectors, `objc_msgSend`, `NSObject` |
| CoreFoundation | the C layer Foundation is built on                           |
| Foundation     | the everyday classes                                         |
