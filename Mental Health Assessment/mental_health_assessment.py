import tkinter as tk
from tkinter import messagebox
import cv2
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageTk

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)

def evaluate_stress(stress):
  if stress<=7:
    return "No"
  elif stress<=9:
    return "Mild"
  elif stress<=12:
    return "Moderate"
  elif stress<=16:
    return "Severe"
  else:
    return "Extreme"

def evaluate_anxiety(anxiety):
  if anxiety<=3:
    return "No"
  elif anxiety<=5:
    return "Mild"
  elif anxiety<=7:
    return "Moderate"
  elif anxiety<=9:
    return "Severe"
  else:
    return "Extreme"

def evaluate_depression(depression):
  if depression<=4:
    return "No"
  elif depression<=6:
    return "Mild"
  elif depression<=10:
    return "Moderate"
  elif depression<=13:
    return "Severe"
  else:
    return "Extreme"

def final_assessment(score):
  if score<=20:
    return "Normal", "Focus on self care"
  elif score<=32:
    return "Mild", "Advise to consult a psychologist"
  elif score<=41:
    return "Moderate", "Best to consult a psychologist"
  elif score<=47:
    return "Severe", "Advise to consult a psychiatrist"
  else:
    return "Extreme", "Must see a psychiatrist"

def assessment(emotions, scores, negatives):

    stress= [1, 6, 8, 11, 12, 14, 18]
    anxiety= [2, 4, 7, 9, 15, 19, 20]
    depression= [3, 5, 10, 13, 16, 17, 21]
    stress_score, anxiety_score, depression_score= 0, 0, 0
    dass_val= []
    for i in range(len(scores)):
        if (i+1) in stress:
            stress_score+= scores[i]
        elif (i+1) in anxiety:
            anxiety_score+= scores[i]
        elif (i+1) in depression:
            depression_score+= scores[i]

    dass_val.append(stress_score)
    dass_val.append(anxiety_score)
    dass_val.append(depression_score)

    total_score= stress_score+anxiety_score+depression_score

    for emo in emotions:
        if emo in negatives:
          total_score+= 1
        else:
           total_score-= 1

    symptom, final_assess= final_assessment(total_score)


    return symptom, final_assess, dass_val, total_score

num_classes = 7
model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
model.load_state_dict(torch.load("emotion_recognition_model.pth", map_location=torch.device('cpu')))
model.eval()

dass21_questions = [
    "I found it hard to wind down",
    "I was aware of dryness of my mouth",
    "I couldn’t seem to experience any positive feeling at all",
    "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion)",
    "I found it difficult to work up the initiative to do things",
    "I tended to over-react to situations",
    "I experienced trembling (e.g. in the hands)",
    "I felt that I was using a lot of nervous energy",
    "I was worried about situations in which I might panic and make a fool of myself",
    "I felt that I had nothing to look forward to",
    "I found myself getting agitated",
    "I found it difficult to relax",
    "I felt down-hearted and blue",
    "I was intolerant of anything that kept me from getting on with what I was doing",
    "I felt I was close to panic",
    "I was unable to become enthusiastic about anything",
    "I felt I wasn’t worth much as a person",
    "I felt that I was rather touchy",
    "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)",
    "I felt scared without any good reason",
    "I felt that life was meaningless"
  ]


class EmotionAssessmentApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Emotion Assessment")

        self.create_start_screen()

    def create_start_screen(self):
        self.start_frame = tk.Frame(self.master)
        self.start_frame.grid(row=0, column=0)

        bold_font = ("", 12, "bold")

        self.start_label = tk.Label(self.start_frame, text="Welcome to Emotion Assessment", font= bold_font)
        self.start_label.pack()

        self.additional_text= tk.Label(self.start_frame, text="""Please read each statement and choose a number 0, 1, 2 or 3 which indicates how much the statement applied to you over the past week. There are no right or wrong answers. Do not spend too much time on any statement.""")
        self.additional_text.pack(anchor= "w")

        self.additional_text1= tk.Label(self.start_frame, text="""\nThe rating scale is as follows:\n
0\tDid not apply to me at all\n
1\tApplied to me to some degree, or some of the time\n
2\tApplied to me to a considerable degree or a good part of time\n
3\tApplied to me very much or most of the time\n\n\n""", font= ("", 12, ""), justify= "left")
        self.additional_text1.pack(anchor= "w")

        self.additional_text2= tk.Label(self.start_frame, text="\n\n\nPlease click 'Start Assessment' to begin.\n", font= bold_font)
        self.additional_text2.pack()
        
        self.start_button= tk.Button(self.start_frame, text="Start Assessment", command=self.start_assessment)
        self.start_button.pack()

        self.additional_text3= tk.Label(self.start_frame, text="\n")
        self.additional_text3.pack()


    def start_assessment(self):
        self.start_frame.destroy()

        self.question_index = 0
        self.emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        self.negative_emotions = ["Angry", "Disgust", "Fear", "Sad"]
        self.emotions = []
        self.scores = []

        self.create_widgets()

    def create_widgets(self):

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video source.")
            return
    
        self.question_label = tk.Label(self.master, text="Question:")
        self.question_label.grid(row=0, column=0)

        self.question_text = tk.Text(self.master, height=2, width=100)
        self.question_text.grid(row=0, column=1, columnspan=2)

        self.score_label = tk.Label(self.master, text="Enter your score (0-3):")
        self.score_label.grid(row=1, column=0)

        self.score_entry = tk.Entry(self.master)
        self.score_entry.grid(row=1, column=1)

        self.next_button = tk.Button(self.master, text="Next", command=self.next_question)
        self.next_button.grid(row=1, column=2)

        self.video_label = tk.Label(self.master, text="Video Feed")
        self.video_label.grid(row=2, column=0, columnspan=3)

        self.video_frame = tk.Label(self.master)
        self.video_frame.grid(row=3, column=0, columnspan=3)

        self.display_next_question()



    def display_next_question(self):
        
        ret, frame = self.cap.read()
        if ret:
            input_frame = preprocess_frame(frame)

            with torch.no_grad():
                output = model(input_frame)

            predicted_class_index = torch.argmax(output, dim=1).item()
            emotion = self.emotion_labels[predicted_class_index]
            self.emotions.append(emotion)

            self.video_frame.img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_frame.config(image=self.video_frame.img)

        if self.question_index < len(dass21_questions):
            self.question_text.delete(1.0, tk.END)
            self.question_text.insert(tk.END, dass21_questions[self.question_index])

            self.question_index += 1
        else:
            self.finish_assessment()


    def next_question(self):
        score = self.score_entry.get()
        if score.isdigit() and 0 <= int(score) <= 3:
            score = int(score)
            self.scores.append(score)
            self.score_entry.delete(0, tk.END)
            self.display_next_question()
            
        else:
            messagebox.showerror("Error", "Please enter a valid score (0-3).")

    def finish_assessment(self):
        
        self.question_label.destroy()
        self.question_text.destroy()
        self.video_label.destroy()
        self.video_frame.destroy()
        self.score_label.destroy()
        self.score_entry.destroy()
        self.next_button.destroy()

        symptom, final_assess, dass_val, total_score = assessment(self.emotions, self.scores, self.negative_emotions)
        stress_val = evaluate_stress(dass_val[0])
        anxiety_val = evaluate_anxiety(dass_val[1])
        depression_val = evaluate_depression(dass_val[2])

        self.final_assessment_window = tk.Toplevel(self.master)
        self.final_assessment_window.title("Final Assessment")
        self.final_assessment_window.geometry("800x400")

        bold_font1= ("", 12, "bold")
        bold_font2= ("", 10, "bold")
        bold_font3= ("", 14, "bold")

        self.final_assessment_label1 = tk.Label(self.final_assessment_window, text="This is not a professional diagnosis, consult a doctor for expert opinion.", font= bold_font2)
        self.final_assessment_label1.pack()

        self.final_assessment_label2 = tk.Label(self.final_assessment_window, text="Psychological Assessment", font=("", 18, "bold"))
        self.final_assessment_label2.pack()

        self.final_assessment_label3= tk.Label(self.final_assessment_window, text=f"DASS Score: {total_score}", font= bold_font1, anchor= "w", justify= "left")
        self.final_assessment_label3.pack(fill= "x")

        self.final_assessment_label4= tk.Label(self.final_assessment_window, text=f"Symptoms: You appear to have '{stress_val}' stress, '{anxiety_val}' anxiety and '{depression_val}' depression.\n", font= ("", 12, ""), wraplength=700, anchor= "w", justify= "left")
        self.final_assessment_label4.pack(fill= "x")

        self.final_assessment_label5= tk.Label(self.final_assessment_window, text="Final Assessment:", font= bold_font3, anchor= "w", justify= "left")
        self.final_assessment_label5.pack(fill= "x")

        self.final_assessment_label6= tk.Label(self.final_assessment_window, text=f"Assessment- {symptom}\nSuggestion- {final_assess}.", font= ("", 12, ""), anchor= "w", justify= "left")
        self.final_assessment_label6.pack(fill= "x")

        self.final_assessment_label7= tk.Label(self.final_assessment_window, text=f"\nEmotions throughout the test: {', '.join(self.emotions)}", anchor='w', wraplength=700, justify= "left")
        self.final_assessment_label7.pack(fill= "x")


        self.final_assessment_window.protocol("WM_DELETE_WINDOW", self.return_to_start_assessment)

    def return_to_start_assessment(self):

        self.final_assessment_window.destroy()


        self.create_start_screen()



def main():
    root = tk.Tk()
    app = EmotionAssessmentApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
