# Sleep "forever" in the background. 
sleep infinity&

# Take note of the 'sleeper' process ID
sleeper_pid=$!

# Create a function that gets executed when this main process receives signal TERM
shutdown_pod(){
  1>&2 echo Exiting main process! 
  kill $sleeper_pid
  exit 0
}
trap shutdown_pod TERM

# Wait for the sleeper process to finish. This is important!
wait
 