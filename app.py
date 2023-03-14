from flask import Flask, flash, render_template, url_for, redirect, request, session
from flask_login import login_user, LoginManager, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
from models import db, User, Session
from forms import SignUpForm, LoginForm
import os


app = Flask(__name__)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.permanent_session_lifetime = timedelta(minutes=60)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
db.app = app
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
# get id from session,then retrieve user object from database with peewee query
def load_user(user_id):
    return User.query.get(int(user_id))

#Displayed if user attempts to modify URL to access page without being logged in.
login_manager.login_message = u"You must login to access this page!"

#Home route, displays Buttons routed to Login & Sign Up.
@app.route('/')
def home():
    return render_template('home.html')

#Displays login page and appropriate responses for invalid login attempts.
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                session.permanent = True
                return redirect(url_for('dashboard'))
            else: flash('Incorrect Password!')
            return render_template('login.html', form=form)
        else:
            flash('Email not in database!')
            return render_template('login.html', form=form)
    return render_template('login.html', form=form)


#Logs out user.
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

#Displays signup page and appropriate response if provided email is already in use.
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignUpForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(email=form.email.data,username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)

#Displays dashboard to logged in user, user uploads image on this page and gets results.
#Clicking of submit button in dashboard.html initiates use of route.
@app.route('/dashboard', methods=['POST','GET'])
@login_required
def dashboard():
    if request.method=='POST':
        file_hold=request.files['image']
        filename=file_hold.filename
        #Building of user folder path in static/uploads to store submitted jpg files.
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], current_user.email)
        #Creation of user folder path only if it doesn't already exist.
        if not os.path.exists(user_folder):
           os.makedirs(user_folder)
        #--------------------------------------------------------------------------
        #Saving of submitted file to user folder path.
        filename2 = "/".join([current_user.email, filename])
        Path=os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        file_hold.save(Path)
        #--------------------------------------------------------
        userr = User.query.filter_by(email=current_user.email).first()
        now = datetime.now()
        dt_string = now.strftime("%Y/%m/%d %H:%M %p")
        session = Session(date=dt_string,prediction='',disease='',treatment='',image=filename,user=userr)
        db.session.add(session)
        db.session.commit()
        return render_template('index.html',upload_hold=True,img_name=filename)
    return render_template('index.html',upload_hold=False,img_name="")


#Displays aboutus route to logged in user.
@app.route('/aboutus')
@login_required
def go_to_aboutus():
    return render_template('aboutus.html')

#Displays history route to logged in user.
@app.route('/history')
@login_required
def go_to_history():
    page = request.args.get('page', 1, type=int)
    user = User.query.filter_by(email=current_user.email).first()
    #Querying of session model to only return the currents user sessions by id.
    #The sorting of the sessions in descending order, the most recent upload will be displayed first.
    #The use of pagination to display 3 post per page.
    sessions = Session.query.filter_by(user=user).order_by(Session.id.desc()).paginate(page=page, per_page=3)
    return render_template('history.html',sessions=sessions)

#Route deletes upload from history page.
#Clicking of delete button in history.html initiates use of route.
@app.route('/delete/<int:id>')
@login_required
def delete_from_history(id):
    session = Session.query.get(id)
    db.session.delete(session)
    db.session.commit()
    return redirect('/history')

#Displays profile route to logged in user.
@app.route('/profile')
@login_required
def go_to_profile():
    return render_template('profile.html')

#Displays plants route to logged in user.
@app.route('/plants')
@login_required
def go_to_plants():
    return render_template('plants.html')

#Displays diseases route to logged in user.
@app.route('/diseases')
@login_required
def go_to_diseases():
    return render_template('diseases.html')


if __name__ == "__main__":
    #Creates and initalizes database. 
    #Only creates database if database.db is missing,
    #if not the existing database.db will be used.
    db.create_all()
    app.run(port=8000, debug=True)
