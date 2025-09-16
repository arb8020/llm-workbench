todo

hell bench: 

spike — 9:46 PM
which begs the question: does rl bias for the 'reward seeking' feature, and if so, is there a way to design an rl experiment that doesn't do this?
arb8020 — 9:46 PM
maybe auxiliary rewards are the problem
maybe rewards are the problem
spike — 9:47 PM
easily resolvable question: can the llm figure out what reward functions are present throughout a run?
gut tells me yes
arb8020 — 9:47 PM
i actually have that one on my list
i was wondering if you put a LLM into multiple different envs at random if it could get stable training by figuring out what the reward fn is for a given env
gut says yes
spike — 9:48 PM
as in, the task is to figure out the reward function?
or is that your job?
arb8020 — 9:48 PM
yeah implicitly the task is to figure out the reward

env a with toolset a has some reward (call 'complete' when the calculator shows some state)
env b with toolset b has some reward (call 'complete' after writing x amount of words)
or you could even allow both toolsets a and b, and have the model figure out which env its in through some other clues, like sys prompt or user language/tool responses 
spike — 9:50 PM
would you inform the model it had these tools?
arb8020 — 9:51 PM
yeah it gets tools=[tools] in each client.messages.create right
spike — 9:51 PM
you could technically turn that off and have it figure out its moveset too
arb8020 — 9:51 PM
cursed but funny
spike — 9:51 PM
i mean this whole experiment description feels like something kafka would dream up
arb8020 — 9:52 PM
kafka could be my middle name
spike — 9:52 PM
'you awake with nothing but a stack of chainsaws, a handgun, and are told to do "something". you don't know what that thing is. but apparently, you cannot leave until you do it properly'
arb8020 — 9:52 PM
ok i will write this up more formally later tn
death kicks sword across the floor
'pick it up.'
arb8020 — 9:52 PM
jigsaw coded
spike — 9:53 PM
introducing hell bench
