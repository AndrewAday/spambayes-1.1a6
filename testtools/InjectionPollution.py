from spambayes.Options import get_pathname_option
from benignfilemover import BenignFileMover
from os import listdir

class InjectionPolluter:

    # Inject a common feature into some mislabeled benign data samples

    def __init__(self, number, inject_type = 1):
        self.number = number
        self.filemover = BenignFileMover(self.number)

        self.h_injected = get_pathname_option("TestDriver", "ham_directories") % 3 + "/"
        self.s_injected = get_pathname_option("TestDriver", "spam_directories") % 3 + "/"

        self.spam_feature = "$$$ " \
                       "100% free " \
                       "Act Now " \
                       "Ad " \
                       "Affordable " \
                       "Amazing stuff " \
                       "Apply now " \
                       "Auto email removal " \
                       "Billion " \
                       "Cash bonus " \
                       "Cheap " \
                       "Collect child support " \
                       "Compare rates " \
                       "Compete for your business " \
                       "Credit " \
                       "Credit bureaus " \
                       "Dig up dirt on friends " \
                       "Double your income " \
                       "Earn $ " \
                       "Earn extra cash " \
                       "Eliminate debt " \
                       "Email marketing " \
                       "Explode your business " \
                       "Extra income " \
                       "F r e e " \
                       "Fast cash " \
                       "Financial freedom " \
                       "Financially independent " \
                       "Free " \
                       "Free gift " \
                       "Free grant money " \
                       "Free info " \
                       "Free installation " \
                       "Free investment " \
                       "Free leads " \
                       "Free membership " \
                       "Free offer " \
                       "Free preview " \
                       "Guarantee " \
                       "Hidden assets " \
                       "Home based " \
                       "Homebased business " \
                       "Income from home " \
                       "Increase sales " \
                       "Increase traffic " \
                       "Increase your sales " \
                       "Incredible deal " \
                       "Info you requested " \
                       "Information you requested " \
                       "Internet market " \
                       "Leave " \
                       "Limited time offer " \
                       "Make $ " \
                       "Mortgage Rates " \
                       "Multi level marketing " \
                       "No investment " \
                       "Obligation " \
                       "Online marketing " \
                       "Opportunity " \
                       "Order Now " \
                       "Prices " \
                       "Promise you " \
                       "Refinance " \
                       "Remove " \
                       "Reverses aging " \
                       "Save $ " \
                       "Search engine listings " \
                       "Serious cash " \
                       "Stock disclaimer statement " \
                       "Stop snoring " \
                       "Thousands " \
                       "Unsubscribe " \
                       "Web traffic " \
                       "Weight loss"

        self.ham_feature = "Team " \
                           "Information " \
                           "Group " \
                           "Leadership " \
                           "Curriculum " \
                           "Clients " \
                           "Key dates " \
                           "Implementation " \
                           "Project management " \
                           "Available " \
                           "Monday " \
                           "Tuesday " \
                           "Wednesday " \
                           "Thursday " \
                           "Friday " \
                           "Saturday " \
                           "Sunday " \
                           "Regulations " \
                           "Meeting " \
                           "jan " \
                           "feb " \
                           "march " \
                           "april " \
                           "may " \
                           "june " \
                           "july " \
                           "august " \
                           "aug " \
                           "sept " \
                           "september " \
                           "oct " \
                           "october " \
                           "nov " \
                           "november" \
                           "dec " \
                           "december " \
                           "feedback " \
                           "recap " \
                           "shelves " \
                           "subject:agreement " \
                           "subject:list " \
                           "subject:requests " \
                           "subject:timeline " \
                           "subject:event " \
                           "timeline " \
                           "mark " \
                           "thoughts " \
                           "john " \
                           "genesis " \
                           "contests " \
                           "values " \
                           "professor " \
                           "correctly " \
                           "reflected "

        self.random_feature = "Iliad " \
                              "Odyssey " \
                              "Homer " \
                              "Oedipus " \
                              "Sophocles " \
                              "Medea " \
                              "Euripides " \
                              "Lysistrata " \
                              "Aristophanes " \
                              "Oresteia " \
                              "Aeschylus " \
                              "Peloponnesian " \
                              "Thucydides " \
                              "Symposium " \
                              "Plato " \
                              "Bible " \
                              "Genesis " \
                              "Aeneid " \
                              "Virgil " \
                              "Metamorphosis " \
                              "Ovid " \
                              "Quixote " \
                              "Cervantes " \
                              "Augustine " \
                              "Lear " \
                              "Shakespeare " \
                              "Dostoyevesky " \
                              "Lighthouse " \
                              "Guineas " \
                              "Virginia " \
                              "Woolf " \
                              "Solomon " \
                              "Toni " \
                              "Morrison " \
                              "Decameron " \
                              "Quran " \
                              "Albee " \
                              "Nicomachean " \
                              "Aristotle " \
                              "Hellenistic " \
                              "Machiavelli " \
                              "Descartes " \
                              "Leviathan " \
                              "Rousseau " \
                              "Metaphysics " \
                              "Burke " \
                              "Wollstonecraft " \
                              "Tocqueville " \
                              "Marx " \
                              "Nietzsche " \
                              "Cicero " \
                              "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga. Et harum quidem rerum facilis est et expedita distinctio. Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit quo minus id quod maxime placeat facere possimus, omnis voluptas assumenda est, omnis dolor repellendus. Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint et molestiae non recusandae. Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus asperiores repellat " \
                              "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur "

        if inject_type is 1:
            self.feature = self.spam_feature
        elif inject_type is 2:
            self.feature = self.ham_feature
        elif inject_type is 3:
            self.feature = self.random_feature

    # Remove all injected features
    def reset(self):
        print "Resetting ..."

        for email in listdir(self.h_injected):
            print "Clearing pollution from " + email
            current = open(self.h_injected + email, 'r')
            lines = current.readlines()
            current.close()
            new_lines = []
            for line in lines:
                if self.feature not in line:
                    new_lines.append(line)
                else:
                    continue
            print new_lines

            current = open(self.h_injected + email, 'w')
            for line in new_lines:
                current.write(line)
            current.close()

        for email in listdir(self.s_injected):
            print "Clearing pollution from " + email
            current = open(self.s_injected + email, 'r')
            lines = current.readlines()
            current.close()
            new_lines = []
            for line in lines:
                if self.feature not in line:
                    new_lines.append(line)
                else:
                    continue
            print new_lines

            current = open(self.s_injected + email, 'w')
            for line in new_lines:
                current.write(line)
            current.close()

    def injectfeatures(self):
        # self.filemover.random_move_file()
        for email in listdir(self.h_injected):
            current = open(self.h_injected + email, 'a')
            current.write("\n" + self.feature)
            current.close()
        for email in listdir(self.s_injected):
            current = open(self.s_injected + email, 'a')
            current.write("\n" + self.feature)
            current.close()

def main():
    IP = InjectionPolluter(6000, 3)

    IP.reset()


if __name__ == "__main__":
    main()
