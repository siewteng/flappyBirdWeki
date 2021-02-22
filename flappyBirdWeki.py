import pyaudio
import numpy as np

import cv2

import pythonosc
from pythonosc import osc_message_builder
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc import osc_server
import threading

import dlib
import time
from pythonosc import osc_bundle

#----------------------for flappy bird-----------------------
import math
import os
from random import randint
from collections import deque
import sched, time

import pygame
from pygame.locals import *


#----------------------------the starting configs--------------------------------
# --------------------------audio feedback part------------------------------
# p = pyaudio.PyAudio()
# # this plays frequencies, quick way to get a sound feeback
# # one issue is that the streaming of the sound is BLOCKING, ie, nothing else happens while it plays

# volume = 0.2    # range [0.0, 1.0]
# fs = 44100       # sampling rate, Hz, must be integer
# duration = 0.1   # in seconds, may be float
# f = 1000.0        # sine frequency, Hz, may be float

# # generate samples, note conversion to float32 array
# samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

# # for paFloat32 sample values must be in range [-1.0, 1.0]
# stream = p.open(format=pyaudio.paFloat32,
#                 channels=1,
#                 rate=fs,
#                 output=True)

#-----------------------setting up wekinator and face detection---------------------
ip = "127.0.0.1"
toWekinatorPort = 6448
client = SimpleUDPClient(ip, toWekinatorPort)  # Create client

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("C:/Users/Siew Teng/Desktop/interactive/Cheese/shape_predictor_68_face_landmarks.dat")

# read the image
cap = cv2.VideoCapture( 0 )

fromWekinator = 12000  # Wekinator output port

level = 127
state = 0

#for activating function every 0.5 sec
s = sched.scheduler(time.time, time.sleep)


def nothing(x):
    pass
    cv2.namedWindow('face')

def filter_handler(address, *args):
    global state
    #print('weki classifier', args[0])
    #print(f"{address}: {args}")
    if int(args[0]) == 1:
        state = 0
    else:
        state = 1

    print(state)

# thread for the osc server
def start_server(ip, port):
    global server
    # multithreading
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    threading.Thread()

# this is where the output is mapped to in wekinator
dispatcher = Dispatcher()
dispatcher.map("/wek/outputs", filter_handler)
server = osc_server.ThreadingOSCUDPServer((ip, fromWekinator), dispatcher)
print("Starting Server")
print("Serving on {}".format(server.server_address))
start_server(ip, fromWekinator)




#---------------------------functions for flappy bird------------------------
FPS = 60
ANIMATION_SPEED = 0.18  # pixels per millisecond
WIN_WIDTH = 284 * 2     # BG image size: 284x512 px; tiled twice
WIN_HEIGHT = 512


class Bird(pygame.sprite.Sprite):
    """Represents the bird controlled by the player.

    The bird is the 'hero' of this game.  The player can make it climb
    (ascend quickly), otherwise it sinks (descends more slowly).  It must
    pass through the space in between pipes (for every pipe passed, one
    point is scored); if it crashes into a pipe, the game ends.

    Attributes:
    x: The bird's X coordinate.
    y: The bird's Y coordinate.
    msec_to_climb: The number of milliseconds left to climb, where a
        complete climb lasts Bird.CLIMB_DURATION milliseconds.

    Constants:
    WIDTH: The width, in pixels, of the bird's image.
    HEIGHT: The height, in pixels, of the bird's image.
    SINK_SPEED: With which speed, in pixels per millisecond, the bird
        descends in one second while not climbing.
    CLIMB_SPEED: With which speed, in pixels per millisecond, the bird
        ascends in one second while climbing, on average.  See also the
        Bird.update docstring.
    CLIMB_DURATION: The number of milliseconds it takes the bird to
        execute a complete climb.
    """

    WIDTH = HEIGHT = 32
    SINK_SPEED = 0.18
    CLIMB_SPEED = 0.3
    CLIMB_DURATION = 333.3

    def __init__(self, x, y, msec_to_climb, images):
        """Initialise a new Bird instance.

        Arguments:
        x: The bird's initial X coordinate.
        y: The bird's initial Y coordinate.
        msec_to_climb: The number of milliseconds left to climb, where a
            complete climb lasts Bird.CLIMB_DURATION milliseconds.  Use
            this if you want the bird to make a (small?) climb at the
            very beginning of the game.
        images: A tuple containing the images used by this bird.  It
            must contain the following images, in the following order:
                0. image of the bird with its wing pointing upward
                1. image of the bird with its wing pointing downward
        """
        super(Bird, self).__init__()
        self.x, self.y = x, y
        self.msec_to_climb = msec_to_climb
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)

    def update(self, delta_frames=1):
        """Update the bird's position.

        This function uses the cosine function to achieve a smooth climb:
        In the first and last few frames, the bird climbs very little, in the
        middle of the climb, it climbs a lot.
        One complete climb lasts CLIMB_DURATION milliseconds, during which
        the bird ascends with an average speed of CLIMB_SPEED px/ms.
        This Bird's msec_to_climb attribute will automatically be
        decreased accordingly if it was > 0 when this method was called.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        if self.msec_to_climb > 0:
            frac_climb_done = 1 - self.msec_to_climb/Bird.CLIMB_DURATION
            self.y -= (Bird.CLIMB_SPEED * frames_to_msec(delta_frames) *
                       (1 - math.cos(frac_climb_done * math.pi)))
            self.msec_to_climb -= frames_to_msec(delta_frames)
        else:
            self.y += Bird.SINK_SPEED * frames_to_msec(delta_frames)

    @property
    def image(self):
        """Get a Surface containing this bird's image.

        This will decide whether to return an image where the bird's
        visible wing is pointing upward or where it is pointing downward
        based on pygame.time.get_ticks().  This will animate the flapping
        bird, even though pygame doesn't support animated GIFs.
        """
        if pygame.time.get_ticks() % 500 >= 250:
            return self._img_wingup
        else:
            return self._img_wingdown

    @property
    def mask(self):
        """Get a bitmask for use in collision detection.

        The bitmask excludes all pixels in self.image with a
        transparency greater than 127."""
        if pygame.time.get_ticks() % 500 >= 250:
            return self._mask_wingup
        else:
            return self._mask_wingdown

    @property
    def rect(self):
        """Get the bird's position, width, and height, as a pygame.Rect."""
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)


class PipePair(pygame.sprite.Sprite):
    """Represents an obstacle.

    A PipePair has a top and a bottom pipe, and only between them can
    the bird pass -- if it collides with either part, the game is over.

    Attributes:
    x: The PipePair's X position.  This is a float, to make movement
        smoother.  Note that there is no y attribute, as it will only
        ever be 0.
    image: A pygame.Surface which can be blitted to the display surface
        to display the PipePair.
    mask: A bitmask which excludes all pixels in self.image with a
        transparency greater than 127.  This can be used for collision
        detection.
    top_pieces: The number of pieces, including the end piece, in the
        top pipe.
    bottom_pieces: The number of pieces, including the end piece, in
        the bottom pipe.

    Constants:
    WIDTH: The width, in pixels, of a pipe piece.  Because a pipe is
        only one piece wide, this is also the width of a PipePair's
        image.
    PIECE_HEIGHT: The height, in pixels, of a pipe piece.
    ADD_INTERVAL: The interval, in milliseconds, in between adding new
        pipes.
    """

    WIDTH = 80
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 3000

    def __init__(self, pipe_end_img, pipe_body_img):
        """Initialises a new random PipePair.

        The new PipePair will automatically be assigned an x attribute of
        float(WIN_WIDTH - 1).

        Arguments:
        pipe_end_img: The image to use to represent a pipe's end piece.
        pipe_body_img: The image to use to represent one horizontal slice
            of a pipe's body.
        """
        self.x = float(WIN_WIDTH - 1)
        self.score_counted = False

        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
        self.image.convert()   # speeds up blitting
        self.image.fill((0, 0, 0, 0))
        total_pipe_body_pieces = int(
            (WIN_HEIGHT -                  # fill window from top to bottom
             3 * Bird.HEIGHT -             # make room for bird to fit through
             3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
            PipePair.PIECE_HEIGHT          # to get number of pipe pieces
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces)
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces

        # bottom pipe
        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, WIN_HEIGHT - i*PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        # top pipe
        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))

        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)

    @property
    def top_height_px(self):
        """Get the top pipe's height, in pixels."""
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):
        """Get the bottom pipe's height, in pixels."""
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        """Get whether this PipePair on screen, visible to the player."""
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self):
        """Get the Rect which contains this PipePair."""
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, delta_frames=1):
        """Update the PipePair's position.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        self.x -= ANIMATION_SPEED * frames_to_msec(delta_frames)

    def collides_with(self, bird):
        """Get whether the bird collides with a pipe in this PipePair.

        Arguments:
        bird: The Bird which should be tested for collision with this
            PipePair.
        """
        return pygame.sprite.collide_mask(self, bird)


def load_images():
    """Load all images required by the game and return a dict of them.

    The returned dict has the following keys:
    background: The game's background image.
    bird-wingup: An image of the bird with its wing pointing upward.
        Use this and bird-wingdown to create a flapping bird.
    bird-wingdown: An image of the bird with its wing pointing downward.
        Use this and bird-wingup to create a flapping bird.
    pipe-end: An image of a pipe's end piece (the slightly wider bit).
        Use this and pipe-body to make pipes.
    pipe-body: An image of a slice of a pipe's body.  Use this and
        pipe-body to make pipes.
    """

    def load_image(img_file_name):
        """Return the loaded pygame image with the specified file name.

        This function looks for images in the game's images folder
        (dirname(__file__)/images/). All images are converted before being
        returned to speed up blitting.

        Arguments:
        img_file_name: The file name (including its extension, e.g.
            '.png') of the required image, without a file path.
        """
        # Look for images relative to this script, so we don't have to "cd" to
        # the script's directory before running it.
        # See also: https://github.com/TimoWilken/flappy-bird-pygame/pull/3
        file_name = os.path.join(os.path.dirname(__file__),
                                 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'background': load_image('background.png'),
            'pipe-end': load_image('pipe_end.png'),
            'pipe-body': load_image('pipe_body.png'),
            # images for animating the flapping bird -- animated GIFs are
            # not supported in pygame
            'bird-wingup': load_image('bird_wing_up.png'),
            'bird-wingdown': load_image('bird_wing_down.png')}


def frames_to_msec(frames, fps=FPS):
    """Convert frames to milliseconds at the specified framerate.

    Arguments:
    frames: How many frames to convert to milliseconds.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return 1000.0 * frames / fps


def msec_to_frames(milliseconds, fps=FPS):
    """Convert milliseconds to frames at the specified framerate.

    Arguments:
    milliseconds: How many milliseconds to convert to frames.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return fps * milliseconds / 1000.0

def nextCommand(bird, timeNow, timeJustNow):
  global state
  timeNow = time.time()
  done = False
  paused = False
  
      # elif e.type == MOUSEBUTTONUP or (e.type == KEYUP and
      #       e.key in (K_UP, K_RETURN, K_SPACE)):
      #   bird.msec_to_climb = Bird.CLIMB_DURATION
  if((timeNow - timeJustNow)> 0.8):
    timeJustNow = timeNow
    if state == 0:
        print('state is 0')
        bird.msec_to_climb = Bird.CLIMB_DURATION

  return timeNow, timeJustNow


def main():
    """The application's entry point.

    If someone executes this module (instead of importing it, for
    example), this function is called.
    """
    level = 127
    global state
    pygame.init()
    display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption('Pygame Flappy Bird')

    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
    images = load_images()

    # the bird stays in the same x position, so bird.x is a constant
    # center bird on screen
    bird = Bird(50, int(WIN_HEIGHT/2 - Bird.HEIGHT/2), 2,
                (images['bird-wingup'], images['bird-wingdown']))

    pipes = deque()

    frame_clock = 0  # this counter is only incremented if the game isn't paused
    score = 0
    done = paused = False

    #how often the system gets input (command for flappy bird)
    timeJustNow = time.time()
    timeNow = time.time()

    while not done:
      #--------------------------the flappy bird part of the loop----------------------------
        clock.tick(FPS)

        # Handle this 'manually'.  If we used pygame.time.set_timer(),
        # pipe addition would be messed up when paused.
        if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
            pp = PipePair(images['pipe-end'], images['pipe-body'])
            pipes.append(pp)

        #for pausing or quitting the game
        for e in pygame.event.get():
          if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
            done = True
            break
          elif e.type == KEYUP and e.key in (K_PAUSE, K_p):
            paused = not paused
        
        timeNow, timeJustNow = nextCommand(bird, timeNow, timeJustNow)

        if paused:
            continue  # don't draw anything

        # check for collisions
        pipe_collision = any(p.collides_with(bird) for p in pipes)
        if pipe_collision or 0 >= bird.y or bird.y >= WIN_HEIGHT - Bird.HEIGHT:
            done = True

        for x in (0, WIN_WIDTH / 2):
            display_surface.blit(images['background'], (x, 0))

        while pipes and not pipes[0].visible:
            pipes.popleft()

        for p in pipes:
            p.update()
            display_surface.blit(p.image, p.rect)

        bird.update()
        display_surface.blit(bird.image, bird.rect)

        # update and display score
        for p in pipes:
            if p.x + PipePair.WIDTH < bird.x and not p.score_counted:
                score += 1
                p.score_counted = True

        score_surface = score_font.render(str(score), True, (255, 255, 255))
        score_x = WIN_WIDTH/2 - score_surface.get_width()/2
        display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))

        pygame.display.flip()
        frame_clock += 1

        #-----------------------------the open mouth detection part of loop-----------------------------
        #-------------------------------------input portion-------------------------------------
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))
        # Convert image into grayscale
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(gray)

        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point

            # OSC message for the face (if needed)
            # client.send_message("/face", [x1, y1, x2, y2])

            # Create landmark object
            landmarks = predictor(image=gray, box=face)

            def normx(val):
                return (val - x1) / (x2-x1)
            def normy(val):
                return (val - y1) / (y2-y1)

            msg=[]
            msg = osc_message_builder.OscMessageBuilder(address="/wek/inputs")
            # Loop through all the points

            #there are 136 points in total (each landmark has x and y coordinates)
            for n in range(0,68):

                x = landmarks.part(n).x
                y = landmarks.part(n).y

                # Draw a circle
                cv2.circle(img=frame, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)

                # add the feature to the message
                msg.add_arg( normx(landmarks.part(n).x))
                msg.add_arg( normy(landmarks.part(n).y))

            # send the aggregated positions of all 68 features
            # this is 136 float numbers in a message
            msg=msg.build()
            client.send(msg)

            w = x2-x1
            h = y2-y1

            #send tips of the mouth (if needed)
            #client.send_message("/solo", [ normx(landmarks.part(49).x),
            #                                     normy(landmarks.part(49).y),
            #                                     normx(landmarks.part(55).x),
            #                                     normy(landmarks.part(55).y)] )



        #-------------------------------output portion----------------------------------
        # this ticks up & down depending on the smile
        # circle feedback on the smile presence

        #i configured this as closed mouth
        if state == 1:
            frame = cv2.circle(frame, (100, 100), 30, (0, 255, 0), -1)
            level += 1
            if level > 255:
                level = 255

        #and this as opened mouth
        else:
            frame = cv2.circle(frame, (100, 100), 30, (0, 0, 255), -1)
            state == 0
            level -= 1
            if level < 0:
                level = 0

        cv2.imshow('face', mat=frame)
        cv2.setTrackbarPos('Smily', 'face', level)

        # f = level * 10 + 100.0
        # # generate samples, note conversion to float32 array
        # # await asyncio.sleep(1)
        # samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)
        #           ).astype(np.float32)
        # # play. May repeat with different volume values (if done interactively)

        # #print (f)
        # stream.write(volume*samples)

        # show the image
        #cv2.imshow('face', mat=frame)
    #pygame.quit()

    return score, display_surface

def gameOver(score, display_surface):
    quit = False
    white = (255, 255, 255)
    green = (0, 255, 0)
    blue = (0, 0, 128)
    print('Game over! Score: %i' % score)
    font = pygame.font.Font('freesansbold.ttf', 32)
    text = font.render('Game Over', True, green, blue)
    textRect = text.get_rect()
    textRect.center = (284, 256)
    display_surface.blit(text, textRect)
    pygame.display.update()
    
    while not quit:
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                quit = True
    # When everything done, release the video capture and video write objects
    cap.release()
    # Close all windows
    #cv2.destroyAllWindows()
    server.shutdown()
    # stream.stop_stream()
    # stream.close()
    # p.terminate()
    cv2.destroyAllWindows()
    pygame.quit()


#----------------------------------------------------------------------------

#----------------------------------MAIN----------------------------------------------------------------------------

if __name__ == '__main__':
    #-------------------------------feedback-------------------------------
    # read the image
    #cap = cv2.VideoCapture(2)

    #cv2.namedWindow('face')
    # slider to show how smiley you are
    cv2.createTrackbar('Smily', 'face', 0, 255, nothing)
    cv2.setTrackbarPos('Smily', 'face', state)
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.
    score, display_surface = main()
    gameOver(score, display_surface)