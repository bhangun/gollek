

gollek-cli % java -jar target/gollek-cli-1.0.0-SNAPSHOT-runner.jar chat --model google-t5/t5-small

  _____       _      _    
 / ____|     | |    | |   
| |  __  ___ | | ___| | __
| | |_ |/ _ \| |/ _ \ |/ /
| |__| | (_) | |  __/   < 
 \_____|\___/|_|\___|_|\_\

Model: google-t5/t5-small
Provider: libtorch
Commands: 'exit' to quit, '/reset' to clear history.
Note: Use '\' at the end of a line for multiline input.
--------------------------------------------------

>>> who are you

Assistant: {"timestamp":"2026-02-15T14:02:08.623661+07:00","sequence":154,"loggerClassName":"org.slf4j.impl.Slf4jLogger","loggerName":"tech.kayys.gollek.engine.inference.InferenceService","level":"ERROR","message":"Streaming inference failed: requestId=8225feb3-90f5-401a-a855-feb5d51f3125","threadName":"main","threadId":3,"mdc":{},"ndc":"","hostName":"aba.local","processName":"/Library/Java/JavaVirtualMachines/graalvm-25.jdk/Contents/Home/bin/java","