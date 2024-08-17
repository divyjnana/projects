let userScore = 0;
let compScore = 0;


const compChoice = () => {
    const options = ['rock', 'paper', 'scissors'];
    let idx = Math.floor(Math.random() * 3);
    return options[idx];
};

let choices = document.querySelectorAll(".box");
let msg=document.querySelector("#msg");
let usersc=document.querySelector("#userscore");
let comsc=document.querySelector("#comscore")

choices.forEach((box) => {
    box.addEventListener("click", () => {
        const userChoice = box.id;
        const computerChoice = compChoice(); 
        playGame(userChoice, computerChoice);
    });
});

const playGame = (user, comp) => {
    console.log("User choice is", user);
    console.log("Computer choice is", comp);

    if (user === comp) {
        console.log("It's a draw");
        msg.innerText = "It was a draw";
    } else if (user === "paper" && comp === "rock") {
        console.log("User wins!");
        msg.innerText = "You win";
        usersc.innerHTML = userScore + 1;
        userScore++;
    } else if (user === "scissors" && comp === "rock") {
        console.log("Computer wins!");
        msg.innerText = "You lose";
        comsc.innerHTML = compScore + 1;
        compScore++;
    } else if (user === "rock" && comp === "paper") {
        console.log("Computer wins!");
        msg.innerText = "You lose";
        comsc.innerHTML = compScore + 1;
        compScore++;
    } else if (user === "paper" && comp === "scissors") {
        console.log("Computer wins!");
        msg.innerText = "You lose";
        comsc.innerHTML = compScore + 1;
        compScore++;
    } else if (user === "rock" && comp === "scissors") {
        console.log("User wins!");
        msg.innerText = "You win";
        usersc.innerHTML = userScore + 1;
        userScore++;
    } else if (user === "scissors" && comp === "paper") {
        console.log("User wins!");
        msg.innerText = "You win";
        usersc.innerHTML = userScore + 1;
        userScore++;
    }

    console.log("User Score:", userScore);
    console.log("Computer Score:", compScore);
};

